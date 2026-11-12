import torch
import torch.nn as nn
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import BatchSampler, SeriesDataset
from net import ConvNet, StageConfig
from exam import exam
import copy
import torch.nn.functional as F


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Wrap the loader in tqdm for a dynamic per-epoch progress bar
    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch [{epoch + 1:02d}/{total_epochs:02d}] (Train)",
        leave=False,
    )

    # Iterate over batches
    for batch_x, batch_y in progress_bar:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Reset the gradients
        optimizer.zero_grad()

        # Run the model and compute backpropagation
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()

        optimizer.step()
        scheduler.step()

        # Compute batch metrics
        running_loss += loss.item() * batch_x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_x.size(0)

        # Update tqdm live metrics
        current_lr = scheduler.get_last_lr()[0]
        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct / total:.3f}",
            lr=f"{current_lr:.2e}",
        )

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            running_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def main() -> int:
    # Training configuration
    train_path = "data/train_split.npz"
    val_path = "data/val_split.npz"
    batch_size = 128
    epochs = 50
    max_lr = 0.004375552246381812
    weight_decay = 7.998469227079363e-05

    # Use GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Init datasets
    train_dataset = SeriesDataset(train_path, is_train=True)
    val_dataset = SeriesDataset(val_path, is_train=False)

    # Init sampler
    train_sampler = BatchSampler(
        labels=train_dataset.labels,
        batch_size=batch_size,
    )

    # Init loaders
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # Init model
    stages = [
        StageConfig(res_kernel_size=5, down_kernel_size=7, num_blocks=1, out_channels=24),
        StageConfig(res_kernel_size=5, down_kernel_size=7, num_blocks=3, out_channels=48),
        StageConfig(res_kernel_size=9, down_kernel_size=-1, num_blocks=4, out_channels=-1),
    ]
    model = ConvNet(
        stem_channels=64,
        stem_kernel_size=5,
        stages=stages,
    ).to(device)

    # Define class weights for the loss function
    class_weights = torch.tensor([
        1.0160834567543418, 0.10118666140028243, 0.17769307654154717, 1.0501353357227972,
        0.8342551958646349, 1.8268189221969722, 2.376399561211156, 2.5987932925248733,
    ]).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03522350778728246)

    # Init optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        weight_decay=weight_decay,
        eps=6.336816125511985e-05,
    )

    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=0.07126090236240201,
        anneal_strategy="cos",
        div_factor=5.192098738100838,
        final_div_factor=7574.835818578024,
        three_phase=False,
    )

    best_val_acc = 0.0
    best_weights: dict[str, torch.Tensor] | None = None

    print(
        f"Starting training on {train_sampler.total_samples} balanced samples "
        f"({train_sampler.class_samples} per class) for {epochs} epochs..."
    )

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
        )

        val_loss, val_acc = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch [{epoch + 1:02d}/{epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
        )

        # Store best weights temporarily
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

    # Restore the best validation state directly from memory
    print("Loading best weights...")
    if best_weights is not None:
        model.load_state_dict(best_weights)

    print("Evaluating model...")
    model.eval()

    # Final evaluation on the validation set after training
    print("Examining model...")
    final_score = exam(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device,
    )

    print(
        f"Training finished. Best Validation Accuracy: {best_val_acc:.4f}\nFinal Score: {final_score:.4f}"
    )

    # Directly export best state to ONNX
    print("Exporting to onnx...")
    onnx_path = "model.onnx"
    dummy_input = torch.randn(1, 4, 128, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
    )
    print(f"Model exported directly to {onnx_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
