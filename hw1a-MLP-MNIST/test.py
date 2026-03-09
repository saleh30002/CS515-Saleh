import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Dict

from parameters import ExperimentConfig, GlobalConfig


@torch.no_grad()
def run_test(
    model: torch.nn.Module,
    exp: ExperimentConfig,
    global_config: GlobalConfig,
    device: torch.device,
) -> Dict:
    """Evaluate the best saved model on the MNIST test set.

    Args:
        model: The neural network.
        exp: ExperimentConfig with batch_size setting.
        global_config: GlobalConfig with data_dir, save_path, and num_classes.
        device: Target device.

    Returns:
        Dict with keys: "accuracy", "test_loss", "per_class_accuracy".
    """
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_ds = datasets.MNIST(global_config.data_dir, train=False, download=True, transform=tf)
    loader  = DataLoader(test_ds, batch_size=exp.batch_size,
                         shuffle=False, num_workers=global_config.num_workers)

    model.load_state_dict(torch.load(global_config.save_path, map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss, correct, n = 0.0, 0, 0
    class_correct = [0] * global_config.num_classes
    class_total   = [0] * global_config.num_classes

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out   = model(imgs)
        loss  = criterion(out, labels)
        preds = out.argmax(1)

        total_loss += loss.item() * imgs.size(0)
        correct    += preds.eq(labels).sum().item()
        n          += imgs.size(0)
        for p, t in zip(preds, labels):
            class_correct[t] += (p == t).item()
            class_total[t]   += 1

    accuracy  = correct / n
    test_loss = total_loss / n
    per_class = {i: class_correct[i] / class_total[i] for i in range(global_config.num_classes)}

    print(f"\n=== Test Results ===")
    print(f"Overall accuracy: {accuracy:.4f}  ({correct}/{n})")
    print(f"Test loss:        {test_loss:.4f}\n")
    for i, acc in per_class.items():
        print(f"  Digit {i}: {acc:.4f}  ({class_correct[i]}/{class_total[i]})")

    return {"accuracy": accuracy, "test_loss": test_loss, "per_class_accuracy": per_class}
