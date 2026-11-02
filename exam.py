import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from time import perf_counter
from competition_api import N_CLASSES

# MACROS
ALPHA = 0.5
LAMBDA = 0.33
P = 2
EPSILON = 0.01

def score(
	class_metrics: dict[int, dict[str, float | int]],
	wrong_penalty: float = LAMBDA,
) -> float:
	"""Calculate the same class geometric-mean score as the official harness."""
	terms = []

	for class_id in range(N_CLASSES):
		metrics = class_metrics[class_id]
		samples = max(int(metrics["samples"]), 1)
		hits = int(metrics["correct"])
		misses = int(metrics["incorrect"])
		term = samples**ALPHA
		term *= (hits - wrong_penalty * misses) / samples
		term *= (hits / samples) ** P
		terms.append(max(EPSILON, term))

	return float(torch.tensor(terms, dtype=torch.float64).log().mean().exp().item())


def exam(
	model: nn.Module,
	dataloader: DataLoader,
	criterion: nn.Module,
	device: torch.device,
	abstention_threshold: float | None = None,
	time_budget_seconds: float = 60.0,
) -> float:
	"""Evaluate sequentially using the official harness scoring semantics."""
	model.eval()
	class_metrics: dict[int, dict[str, float | int]] = {
		class_id: {
			"prediction_time_seconds": 0.0,
			"samples": 0,
			"correct": 0,
			"incorrect": 0,
			"abstentions": 0,
		}
		for class_id in range(N_CLASSES)
	}
	previous_num_threads = torch.get_num_threads()
	torch.set_num_threads(1)
	model_time_seconds = 0.0

	try:
		with torch.no_grad():
			for batch_x, batch_y in dataloader:
				for sample_x, sample_y in zip(batch_x, batch_y):
					if model_time_seconds >= time_budget_seconds:
						break

					sample_x = sample_x.unsqueeze(0).to(device)
					true_class = int(sample_y.item())
					if device.type == "cuda":
						torch.cuda.synchronize(device)
					start_time = perf_counter()
					logits = model(sample_x)
					if device.type == "cuda":
						torch.cuda.synchronize(device)
					elapsed_seconds = perf_counter() - start_time
					if model_time_seconds + elapsed_seconds > time_budget_seconds:
						model_time_seconds = time_budget_seconds
						break
					model_time_seconds += elapsed_seconds

					prediction = int(torch.argmax(logits, dim=1).item())
					confidence = float(torch.softmax(logits, dim=1).amax().item())
					abstained = (
						abstention_threshold is not None
						and confidence < abstention_threshold
					)
					metrics = class_metrics[true_class]
					metrics["prediction_time_seconds"] += elapsed_seconds
					metrics["samples"] += 1
					if abstained:
						metrics["abstentions"] += 1
					elif prediction == true_class:
						metrics["correct"] += 1
					else:
						metrics["incorrect"] += 1

				if model_time_seconds >= time_budget_seconds:
					break
	finally:
		torch.set_num_threads(previous_num_threads)

	if not any(metrics["samples"] for metrics in class_metrics.values()):
		raise ValueError("The evaluation dataloader must contain at least one sample")

	return score(class_metrics)



