import json
import os
import random
import ssl
from dataclasses import asdict
from datetime import datetime

import numpy as np
import torch

from parameters import EXPERIMENTS, Args, ExperimentConfig, GlobalConfig, get_params
from models.MLP import MLP
from train import run_training
from test  import run_test


# Fix for macOS SSL certificate verification error when downloading MNIST
ssl._create_default_https_context = ssl._create_unverified_context

RESULTS_DIR  = "results"
HISTORY_FILE = os.path.join(RESULTS_DIR, "history.json")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_model(exp: ExperimentConfig, global_config: GlobalConfig) -> MLP:
    return MLP(
        experimentConfig=exp,
        input_size=global_config.input_size,
        num_classes=global_config.num_classes,
    )

def save_to_history(
    exp_name: str,
    results: dict,
    exp: ExperimentConfig,
) -> None:
    """Append test results to the JSON history file.

    Each entry records the experiment name, timestamp, accuracy, loss,
    per-class accuracy, and the full ExperimentConfig used.

    Args:
        exp_name: Name of the experiment.
        results: Dict returned by run_test (accuracy, test_loss, per_class_accuracy).
        exp: ExperimentConfig used for the experiment.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)

    entry = {
        "experiment":         exp_name,
        "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy":           round(results["accuracy"], 6),
        "test_loss":          round(results["test_loss"], 6),
        "per_class_accuracy": {str(k): round(v, 6) for k, v in results["per_class_accuracy"].items()},
        "config":             asdict(exp),
    }
    history.append(entry)

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  [history] Results saved → {HISTORY_FILE}")

def run_experiment(
    exp_name: str,
    exp: ExperimentConfig,
    global_config: GlobalConfig,
    args: Args,
    device: torch.device,
) -> None:
    """Run a single experiment: train, test, visualize, and log results.

    Args:
        exp_name: Name of the experiment (key in EXPERIMENTS).
        exp: ExperimentConfig for this experiment.
        global_config: GlobalConfig with fixed settings.
        args: Runtime CLI args.
        device: Target device.
    """
    print(f"\n{'='*60}")
    print(f"  Experiment: {exp_name}")
    print(f"{'='*60}")

    # Per-experiment save path so models don't overwrite each other
    os.makedirs(RESULTS_DIR, exist_ok=True)
    exp_global = GlobalConfig(
        data_dir=global_config.data_dir,
        num_workers=global_config.num_workers,
        input_size=global_config.input_size,
        num_classes=global_config.num_classes,
        seed=global_config.seed,
        log_interval=global_config.log_interval,
        save_path=os.path.join(RESULTS_DIR, f"{exp_name}_best.pth"),
    )

    model = build_model(exp, exp_global).to(device)

    if args.mode in ("train", "both"):
        run_training(model, exp, exp_global, device)

    if args.mode in ("test", "both"):
        results = run_test(model, exp, exp_global, device)
        save_to_history(exp_name, results, exp)


def main() -> None:
    args, global_config = get_params()

    set_seed(global_config.seed)
    print(f"Seed  : {global_config.seed}")

    device = torch.device(
        args.device if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    if args.run_all:
        print(f"\nRunning all {len(EXPERIMENTS)} experiments...")
        for exp_name, exp in EXPERIMENTS.items():
            run_experiment(exp_name, exp, global_config, args, device)
        print(f"\nAll experiments done. History saved to {HISTORY_FILE}")
    else:
        exp = EXPERIMENTS[args.experiment]
        run_experiment(args.experiment, exp, global_config, args, device)


if __name__ == "__main__":
    main()
