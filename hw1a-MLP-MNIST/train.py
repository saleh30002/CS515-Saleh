import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from parameters import ExperimentConfig, GlobalConfig


def get_loaders(exp: ExperimentConfig, global_config: GlobalConfig) -> tuple:
    """Build train and validation DataLoaders for MNIST.

    Args:
        exp: ExperimentConfig with batch_size setting.
        global_config: GlobalConfig with data_dir and num_workers.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(global_config.data_dir, train=True,  download=True, transform=tf)
    val_ds   = datasets.MNIST(global_config.data_dir, train=False, download=True, transform=tf)

    train_loader = DataLoader(train_ds, batch_size=exp.batch_size,
                              shuffle=True,  num_workers=global_config.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=exp.batch_size,
                              shuffle=False, num_workers=global_config.num_workers)
    return train_loader, val_loader


def regularization_loss(model: nn.Module, regularizer: str, reg_coefficient: float) -> torch.Tensor:
    """Compute L1 or L2 regularization loss over all model parameters.

    Args:
        model: The neural network.
        regularizer: "l1", "l2", or None.
        reg_coefficient: Scaling coefficient for the regularization term.

    Returns:
        Scalar regularization loss tensor.
    """
    if regularizer == "l1":
        return reg_coefficient * sum(p.abs().sum() for p in model.parameters())
    if regularizer == "l2":
        return reg_coefficient * sum(p.pow(2).sum() for p in model.parameters())
    return 0.0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    exp: ExperimentConfig,
    device: torch.device,
    log_interval: int,
) -> tuple:
    """Run one training epoch.

    Args:
        model: The neural network.
        loader: Training DataLoader.
        optimizer: Optimizer instance.
        criterion: Loss function.
        exp: ExperimentConfig with regularizer settings.
        device: Target device.
        log_interval: Print progress every N batches.

    Returns:
        Tuple of (avg_loss, accuracy).
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels) + regularization_loss(model, exp.regularizer, exp.reg_coeff)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        n          += imgs.size(0)

        if (batch_idx + 1) % log_interval == 0:
            print(f"  [{batch_idx+1}/{len(loader)}] "
                  f"loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")

    return total_loss / n, correct / n


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Evaluate the model on the validation set.

    Args:
        model: The neural network.
        loader: Validation DataLoader.
        criterion: Loss function.
        device: Target device.

    Returns:
        Tuple of (avg_loss, accuracy).
    """
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.detach().item() * imgs.size(0)
            correct    += out.argmax(1).eq(labels).sum().item()
            n          += imgs.size(0)
    return total_loss / n, correct / n


def custom_scheduler(
    optimizer: torch.optim.Optimizer,
    exp: ExperimentConfig,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Instantiate the LR scheduler from the experiment config.

    Args:
        optimizer: The optimizer to wrap.
        exp: ExperimentConfig with scheduler settings.

    Returns:
        LRScheduler instance or None if scheduler is "none".
    """
    if exp.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=exp.scheduler_step_size, gamma=exp.scheduler_gamma
        )
    if exp.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=exp.epochs
        )
    return None


def run_training(
    model: nn.Module,
    exp: ExperimentConfig,
    global_config: GlobalConfig,
    device: torch.device,
) -> None:
    """Full training loop with validation, early stopping, and LR scheduling.

    Args:
        model: The neural network.
        exp: ExperimentConfig with all training hyperparameters.
        global_config: GlobalConfig with save_path and log_interval.
        device: Target device.
    """
    train_loader, val_loader = get_loaders(exp, global_config)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=exp.lr, weight_decay=exp.weight_decay
    )
    scheduler = custom_scheduler(optimizer, exp)

    best_acc       = 0.0
    best_weights   = None
    patience_count = 0

    for epoch in range(1, exp.epochs + 1):
        print(f"\nEpoch {epoch}/{exp.epochs}")
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, exp, device, global_config.log_interval
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        print(f"  Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc       = val_acc
            best_weights   = copy.deepcopy(model.state_dict())
            patience_count = 0
            torch.save(best_weights, global_config.save_path)
            print(f"  Saved best model (val_acc={best_acc:.4f})")
        else:
            patience_count += 1

        # Early stopping
        if exp.early_stop_patience > 0 and patience_count >= exp.early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    model.load_state_dict(best_weights)
    print(f"\nTraining done. Best val accuracy: {best_acc:.4f}")
