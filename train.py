import torch
import torch.nn as nn
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import BatchSampler, SeriesDataset
from net import ConvNet
from exam import exam


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
    epochs = 40
    max_lr = 1e-3
    weight_decay = 1e-2

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
    model = ConvNet(in_channels=4, num_classes=8).to(device)
    criterion = nn.CrossEntropyLoss()

    # Init optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        weight_decay=weight_decay,
    )

    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=0.20,
        anneal_strategy="cos",
    )

    best_val_acc = 0.0

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

        # Save checkpoint as weights.pt if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "weights.pt")

    # Final evaluation on the validation set after training
    final_score = exam(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device,
    )

    print(f"Training finished. Best Validation Accuracy: {best_val_acc:.4f}\nFinal Score: {final_score:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
