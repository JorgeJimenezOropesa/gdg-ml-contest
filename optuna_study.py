"""Run the Optuna study described by a YAML configuration file.

Example:
    python optuna_study.py --config optuna_config.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from dataset import BatchSampler, SeriesDataset
from exam import exam
from net import ConvNet, StageConfig


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def suggest(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
    """Apply one Optuna distribution declared in YAML."""
    distribution = spec["type"]
    if distribution == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    if distribution == "int":
        return trial.suggest_int(
            name,
            spec["low"],
            spec["high"],
            step=spec.get("step", 1),
            log=spec.get("log", False),
        )
    if distribution == "float":
        return trial.suggest_float(
            name,
            spec["low"],
            spec["high"],
            step=spec.get("step"),
            log=spec.get("log", False),
        )
    raise ValueError(f"Unsupported distribution type for {name}: {distribution}")


def build_model(trial: optuna.Trial, config: dict[str, Any]) -> ConvNet:
    architecture = config["search"]["architecture"]
    n_stages = suggest(trial, "n_stages", architecture["n_stages"])
    stages: list[StageConfig] = []

    for stage_index in range(n_stages):
        is_last = stage_index == n_stages - 1
        stages.append(
            StageConfig(
                res_kernel_size=suggest(
                    trial,
                    f"stage_{stage_index}_res_kernel_size",
                    architecture["stage"]["res_kernel_size"],
                ),
                down_kernel_size=(
                    -1
                    if is_last
                    else suggest(
                        trial,
                        f"stage_{stage_index}_down_kernel_size",
                        architecture["stage"]["down_kernel_size"],
                    )
                ),
                num_blocks=suggest(
                    trial,
                    f"stage_{stage_index}_num_blocks",
                    architecture["stage"]["num_blocks"],
                ),
                out_channels=(
                    -1
                    if is_last
                    else suggest(
                        trial,
                        f"stage_{stage_index}_out_channels",
                        architecture["stage"]["out_channels"],
                    )
                ),
            )
        )

    return ConvNet(
        stem_channels=suggest(trial, "stem_channels", architecture["stem_channels"]),
        stem_kernel_size=suggest(
            trial, "stem_kernel_size", architecture["stem_kernel_size"]
        ),
        stages=stages,
        dense_hidden_dim=suggest(
            trial, "dense_hidden_dim", architecture["dense_hidden_dim"]
        ),
        in_channels=architecture.get("in_channels", 4),
        num_classes=architecture.get("num_classes", 8),
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> None:
    model.train()
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
        scheduler.step()


def make_loaders(config: dict[str, Any], batch_size: int) -> tuple[DataLoader, DataLoader, BatchSampler]:
    data_config = config["data"]
    runtime = config["runtime"]
    train_dataset = SeriesDataset(data_config["train"], is_train=True)
    val_dataset = SeriesDataset(data_config["validation"], is_train=False)
    train_sampler = BatchSampler(train_dataset.labels, batch_size=batch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=runtime["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=runtime["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_sampler


def make_optimizer(
    model: nn.Module, config: dict[str, Any], learning_rate: float, weight_decay: float
) -> torch.optim.Optimizer:
    optimizer_name = config["training"]["optimizer"].lower()
    optimizer_args = {"lr": learning_rate, "weight_decay": weight_decay}
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), **optimizer_args)
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), **optimizer_args)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def make_objective(config: dict[str, Any], device: torch.device):
    def objective(trial: optuna.Trial) -> float:
        seed_everything(config["runtime"]["seed"] + trial.number)
        model = build_model(trial, config).to(device)
        trial.set_user_attr(
            "parameter_count",
            sum(parameter.numel() for parameter in model.parameters()),
        )

        search_training = config["search"]["training"]
        batch_size = suggest(trial, "batch_size", search_training["batch_size"])
        train_loader, val_loader, train_sampler = make_loaders(config, batch_size)
        learning_rate = suggest(trial, "learning_rate", search_training["learning_rate"])
        weight_decay = suggest(trial, "weight_decay", search_training["weight_decay"])
        label_smoothing = suggest(
            trial, "label_smoothing", search_training["label_smoothing"]
        )
        epochs = config["training"]["epochs"]

        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        optimizer = make_optimizer(
            model, config, learning_rate=learning_rate, weight_decay=weight_decay
        )
        scheduler_config = config["training"]["scheduler"]
        if scheduler_config["name"].lower() != "onecycle":
            raise ValueError(f"Unsupported scheduler: {scheduler_config['name']}")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            total_steps=len(train_loader) * epochs,
            pct_start=scheduler_config["pct_start"],
            anneal_strategy=scheduler_config["anneal_strategy"],
        )

        best_score = 0.0
        for epoch in range(epochs):
            train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
            score = exam(
                model,
                val_loader,
                criterion,
                device,
                time_budget_seconds=config["evaluation"]["exam_time_budget"],
            )
            best_score = max(best_score, score)
            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("train_samples_per_epoch", train_sampler.total_samples)
        return best_score

    return objective


def create_sampler(config: dict[str, Any]) -> optuna.samplers.BaseSampler:
    sampler_config = config["study"]["sampler"]
    name = sampler_config["name"].lower()
    options = {key: value for key, value in sampler_config.items() if key != "name"}
    if name == "tpe":
        return optuna.samplers.TPESampler(**options)
    if name == "random":
        return optuna.samplers.RandomSampler(**options)
    if name == "cmaes":
        return optuna.samplers.CmaEsSampler(**options)
    raise ValueError(f"Unsupported sampler: {name}")


def create_pruner(config: dict[str, Any]) -> optuna.pruners.BasePruner:
    pruner_config = config["study"]["pruner"]
    name = pruner_config["name"].lower()
    options = {key: value for key, value in pruner_config.items() if key != "name"}
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner(**options)
    if name == "median":
        return optuna.pruners.MedianPruner(**options)
    if name == "none":
        return optuna.pruners.NopPruner()
    raise ValueError(f"Unsupported pruner: {name}")


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("The YAML root must be a mapping")
    return config


def serialize_trial(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    return {
        "number": trial.number,
        "state": trial.state.name,
        "value": trial.value,
        "params": trial.params,
        "user_attrs": trial.user_attrs,
        "intermediate_values": trial.intermediate_values,
        "datetime_start": trial.datetime_start.isoformat()
        if trial.datetime_start
        else None,
        "datetime_complete": trial.datetime_complete.isoformat()
        if trial.datetime_complete
        else None,
        "duration_seconds": trial.duration.total_seconds()
        if trial.duration
        else None,
    }


def export_study(
    study: optuna.Study,
    config: dict[str, Any],
    config_path: Path,
    device: torch.device,
) -> None:
    """Write a self-contained snapshot of the study for later analysis."""
    output_config = config["output"]
    output_dir = Path(output_config["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = [serialize_trial(trial) for trial in study.trials]
    completed = [trial for trial in study.trials if trial.value is not None]
    best_trial = serialize_trial(study.best_trial)
    summary = {
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_name": study.study_name,
        "study_id": study._study_id,
        "storage": config["study"]["storage"],
        "direction": config["study"]["direction"],
        "device": str(device),
        "trial_count": len(study.trials),
        "completed_count": len(completed),
        "best_value": study.best_value,
        "best_trial_number": study.best_trial.number,
    }

    shutil.copy2(config_path, output_dir / "config.yaml")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "best_trial.json").write_text(
        json.dumps(best_trial, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "trials.json").write_text(
        json.dumps(trials, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "study_system_attrs.json").write_text(
        json.dumps(study.system_attrs, indent=2) + "\n", encoding="utf-8"
    )

    csv_fields = [
        "number",
        "state",
        "value",
        "datetime_start",
        "datetime_complete",
        "duration_seconds",
        "params",
        "user_attrs",
    ]
    with (output_dir / "trials.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()
        for trial in trials:
            writer.writerow(
                {
                    key: json.dumps(trial[key], sort_keys=True)
                    if isinstance(trial[key], (dict, list))
                    else trial[key]
                    for key in csv_fields
                }
            )

    best_config_path = Path(output_config["best_config"])
    best_config_path.write_text(
        json.dumps(
            {
                "value": study.best_value,
                "trial": study.best_trial.number,
                "params": study.best_trial.params,
                "user_attrs": study.best_trial.user_attrs,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config["runtime"]["seed"])

    requested_device = config["runtime"].get("device", "auto")
    device_name = (
        "cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device
    )
    device = torch.device(device_name if device_name != "auto" else "cpu")
    print(f"Using device: {device}")

    study_config = config["study"]
    study = optuna.create_study(
        study_name=study_config["name"],
        storage=study_config["storage"],
        load_if_exists=study_config["load_if_exists"],
        direction=study_config["direction"],
        sampler=create_sampler(config),
        pruner=create_pruner(config),
    )
    study.optimize(
        make_objective(config, device),
        n_trials=study_config["n_trials"],
        timeout=study_config.get("timeout"),
        gc_after_trial=study_config["gc_after_trial"],
    )

    completed_trials = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        raise RuntimeError("The study finished without a completed trial")
    export_study(study, config, args.config, device)
    result = {
        "value": study.best_value,
        "trial": study.best_trial.number,
        "params": study.best_trial.params,
        "user_attrs": study.best_trial.user_attrs,
        "output_directory": config["output"]["directory"],
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())