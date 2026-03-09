import argparse
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GlobalConfig:
    """Fixed settings that do not change across experiments."""
    data_dir:     str = "./data"
    num_workers:  int = 2
    input_size:   int = 784      # 28x28 flattened
    num_classes:  int = 10
    seed:         int = 42
    log_interval: int = 100      # print every N batches
    save_path:    str = "best_model.pth"


@dataclass
class ExperimentConfig:
    """Hyperparameters that vary across experiments. Please define what you would like
    to run when you run the script in the terminal."""
    hidden_sizes: List[int]
    activation:   str            # "Ex: relu" or "gelu"
    use_bn:       bool           # BatchNorm1d before activation
    dropout:      float          # dropout probability (0.0 = disabled)
    regularizer:  Optional[str]  # "l1", "l2", or None
    reg_coeff:    float          # coefficient for L1/L2 term
    lr:           float
    batch_size:   int
    epochs:       int
    weight_decay: float        # L2 via optimizer (separate from reg_coeff)
    scheduler:          str    # Ex: "step", "cosine", or "none"
    scheduler_step_size: int   # used when scheduler="step"
    scheduler_gamma:     float # used when scheduler="step"
    early_stop_patience: int   # 0 = disabled
    bn_after_activation: bool = False  # if True: Linear -> Activation -> BN; default: Linear -> BN -> Activation


@dataclass
class Args:
    """Arguments supplied by the user at runtime via CLI."""
    experiment: Optional[str]  # None when run_all=True
    mode:       str            # "train", "test", or "both"
    device:     str            # "cpu" or "cuda"
    run_all:    bool           # run every experiment sequentially


EXPERIMENTS: dict = {

    "exp1_baseline": ExperimentConfig(
        hidden_sizes=[512, 256],
        activation="relu",
        use_bn=True,
        dropout=0.3,
        regularizer=None,
        reg_coeff=0.0,
        lr=1e-3,
        batch_size=64,
        epochs=10,
        weight_decay=1e-4,
        scheduler="step",
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        early_stop_patience=5,
    ),

    "exp2_gelu": ExperimentConfig(
        hidden_sizes=[512, 256],
        activation="gelu",          # swap activation
        use_bn=True,
        dropout=0.3,
        regularizer=None,
        reg_coeff=0.0,
        lr=1e-3,
        batch_size=64,
        epochs=10,
        weight_decay=1e-4,
        scheduler="step",
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        early_stop_patience=5,
    ),

    "exp3_deep": ExperimentConfig(
        hidden_sizes=[512, 256, 128],  # deeper network
        activation="relu",
        use_bn=True,
        dropout=0.3,
        regularizer=None,
        reg_coeff=0.0,
        lr=1e-3,
        batch_size=64,
        epochs=10,
        weight_decay=1e-4,
        scheduler="step",
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        early_stop_patience=5,
    ),

    "exp4_shallow": ExperimentConfig(
        hidden_sizes=[512],              # shallower network
        activation="relu",
        use_bn=True,
        dropout=0.3,
        regularizer=None,
        reg_coeff=0.0,
        lr=1e-3,
        batch_size=64,
        epochs=10,
        weight_decay=1e-4,
        scheduler="step",
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        early_stop_patience=5,
    ),

    "exp5_wide": ExperimentConfig(
        hidden_sizes=[1024, 512],         # wider network
        activation="relu",
        use_bn=True,
        dropout=0.3,
        regularizer=None,
        reg_coeff=0.0,
        lr=1e-3,
        batch_size=64,
        epochs=10,
        weight_decay=1e-4,
        scheduler="step",
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        early_stop_patience=5,
    ),

    "exp6_narrow": ExperimentConfig(
        hidden_sizes=[256, 128],          # narrower network
        activation="relu",
        use_bn=True,
        dropout=0.3,
        regularizer=None,
        reg_coeff=0.0,
        lr=1e-3,
        batch_size=64,
        epochs=10,
        weight_decay=1e-4,
        scheduler="step",
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        early_stop_patience=5,
    ),

    "exp7_no_dropout": ExperimentConfig(
        hidden_sizes=[512, 256],
        activation="relu",
        use_bn=True,
        dropout=0.0,                       # dropout disabled
        regularizer=None,
        reg_coeff=0.0,
        lr=1e-3,
        batch_size=64,
        epochs=10,
        weight_decay=1e-4,
        scheduler="step",
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        early_stop_patience=5,
    ),

    "exp8_high_dropout": ExperimentConfig(
        hidden_sizes=[512, 256],
        activation="relu",
        use_bn=True,
        dropout=0.6,                       # high dropout
        regularizer=None,
        reg_coeff=0.0,
        lr=1e-3,
        batch_size=64,
        epochs=10,
        weight_decay=1e-4,
        scheduler="step",
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        early_stop_patience=5,
    ),

    "exp9_no_bn": ExperimentConfig(
        hidden_sizes=[512, 256],
        activation="relu",
        use_bn=False,                      # BatchNorm disabled
        dropout=0.3,
        regularizer=None,
        reg_coeff=0.0,
        lr=1e-3,
        batch_size=64,
        epochs=10,
        weight_decay=1e-4,
        scheduler="step",
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        early_stop_patience=5,
    ),

    "exp10_l1_reg": ExperimentConfig(
        hidden_sizes=[512, 256],
        activation="relu",
        use_bn=True,
        dropout=0.3,
        regularizer="l1",                  # L1 regularization
        reg_coeff=1e-4,
        lr=1e-3,
        batch_size=64,
        epochs=10,
        weight_decay=0.0,
        scheduler="step",
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        early_stop_patience=5,
    ),

    "exp11_l2_reg": ExperimentConfig(
        hidden_sizes=[512, 256],
        activation="relu",
        use_bn=True,
        dropout=0.3,
        regularizer="l2",                  # L2 regularization (manual)
        reg_coeff=1e-4,
        lr=1e-3,
        batch_size=64,
        epochs=10,
        weight_decay=0.0,
        scheduler="step",
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        early_stop_patience=5,
    ),

    "exp12_cosine_scheduler": ExperimentConfig(
        hidden_sizes=[512, 256],
        activation="relu",
        use_bn=True,
        dropout=0.3,
        regularizer=None,
        reg_coeff=0.0,
        lr=1e-3,
        batch_size=64,
        epochs=10,
        weight_decay=1e-4,
        scheduler="cosine",                # cosine annealing
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        early_stop_patience=5,
    ),

    "exp13_high_lr": ExperimentConfig(
        hidden_sizes=[512, 256],
        activation="relu",
        use_bn=True,
        dropout=0.3,
        regularizer=None,
        reg_coeff=0.0,
        lr=1e-1,                       # high learning rate
        batch_size=64,
        epochs=10,
        weight_decay=1e-4,
        scheduler="step",
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        early_stop_patience=5,
    ),

    "exp14_low_lr": ExperimentConfig(
        hidden_sizes=[512, 256],
        activation="relu",
        use_bn=True,
        dropout=0.3,
        regularizer=None,
        reg_coeff=0.0,
        lr=1e-4,                       # high learning rate
        batch_size=64,
        epochs=10,
        weight_decay=1e-4,
        scheduler="step",
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        early_stop_patience=5,
    ),

    "exp15_less_epoch": ExperimentConfig(
        hidden_sizes=[512, 256],
        activation="relu",
        use_bn=True,
        dropout=0.3,
        regularizer=None,
        reg_coeff=0.0,
        lr=1e-3,
        batch_size=64,
        epochs=5,                       # fewer epochs
        weight_decay=1e-4,
        scheduler="step",
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        early_stop_patience=5,
    ),

    "exp16_more_epoch": ExperimentConfig(
        hidden_sizes=[512, 256],
        activation="relu",
        use_bn=True,
        dropout=0.3,
        regularizer=None,
        reg_coeff=0.0,
        lr=1e-3,
        batch_size=64,
        epochs=20,                       # more epochs
        weight_decay=1e-4,
        scheduler="step",
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        early_stop_patience=5,
    ),

    "exp17_basic": ExperimentConfig(
        hidden_sizes=[512, 256],
        activation="relu",
        use_bn=False,
        dropout=0.0,
        regularizer=None,
        reg_coeff=0.0,
        lr=1e-3,
        batch_size=64,
        epochs=10,
        weight_decay=1e-4,
        scheduler="none",
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        early_stop_patience=5,
    ),

    "exp18_basic_dropout": ExperimentConfig(
        hidden_sizes=[512, 256],
        activation="relu",
        use_bn=False,
        dropout=0.3,
        regularizer=None,
        reg_coeff=0.0,
        lr=1e-3,
        batch_size=64,
        epochs=10,
        weight_decay=1e-4,
        scheduler="none",
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        early_stop_patience=5,
    ),

    "exp19_basic_scheduler": ExperimentConfig(
        hidden_sizes=[512, 256],
        activation="relu",
        use_bn=False,
        dropout=0.0,
        regularizer=None,
        reg_coeff=0.0,
        lr=1e-3,
        batch_size=64,
        epochs=10,
        weight_decay=1e-4,
        scheduler="step",
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        early_stop_patience=5,
    ),

    "exp20_basic_bn": ExperimentConfig(
        hidden_sizes=[512, 256],
        activation="relu",
        use_bn=True,
        dropout=0.0,
        regularizer=None,
        reg_coeff=0.0,
        lr=1e-3,
        batch_size=64,
        epochs=10,
        weight_decay=1e-4,
        scheduler="none",
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        early_stop_patience=5,
    ),

    "exp29_basic_bn_after": ExperimentConfig(
        hidden_sizes=[512, 256],
        activation="relu",
        use_bn=True,
        dropout=0.0,
        regularizer=None,
        reg_coeff=0.0,
        lr=1e-3,
        batch_size=64,
        epochs=10,
        weight_decay=1e-4,
        scheduler="none",
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        early_stop_patience=5,
        bn_after_activation=True,  # BatchNorm after activation
    ),

    "exp21_basic_l1": ExperimentConfig(
        hidden_sizes=[512, 256],
        activation="relu",
        use_bn=False,
        dropout=0.0,
        regularizer="l1",
        reg_coeff=1e-4,
        lr=1e-3,
        batch_size=64,
        epochs=10,
        weight_decay=0.0,
        scheduler="none",
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        early_stop_patience=5,
    ),

    "exp23_l1_high_coeff": ExperimentConfig(
        hidden_sizes=[512, 256],
        activation="relu",
        use_bn=True,
        dropout=0.3,
        regularizer="l1",
        reg_coeff=1e-3,              # higher coefficient than exp10 (1e-4)
        lr=1e-3,
        batch_size=64,
        epochs=10,
        weight_decay=0.0,
        scheduler="step",
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        early_stop_patience=5,
    ),

    "exp24_l2_high_coeff": ExperimentConfig(
        hidden_sizes=[512, 256],
        activation="relu",
        use_bn=True,
        dropout=0.3,
        regularizer="l2",
        reg_coeff=1e-3,              # higher coefficient than exp11 (1e-4)
        lr=1e-3,
        batch_size=64,
        epochs=10,
        weight_decay=0.0,
        scheduler="step",
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        early_stop_patience=5,
    ),

    "exp22_bn_after_activation": ExperimentConfig(
        hidden_sizes=[512, 256],
        activation="relu",
        use_bn=True,
        dropout=0.3,
        regularizer=None,
        reg_coeff=0.0,
        lr=1e-3,
        batch_size=64,
        epochs=10,
        weight_decay=1e-4,
        scheduler="step",
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        early_stop_patience=5,
        bn_after_activation=True,
    ),

    "exp23_basic_deeper": ExperimentConfig(
        hidden_sizes=[512, 256, 128],
        activation="relu",
        use_bn=False,
        dropout=0.0,
        regularizer=None,
        reg_coeff=0.0,
        lr=1e-3,
        batch_size=64,
        epochs=10,
        weight_decay=1e-4,
        scheduler="none",
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        early_stop_patience=5,
    ),

    "exp24_basic_shallower": ExperimentConfig(
        hidden_sizes=[512],
        activation="relu",
        use_bn=False,
        dropout=0.0,
        regularizer=None,
        reg_coeff=0.0,
        lr=1e-3,
        batch_size=64,
        epochs=10,
        weight_decay=1e-4,
        scheduler="none",
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        early_stop_patience=5,
    ),

     "exp25_basic_narrower": ExperimentConfig(
        hidden_sizes=[256, 128],
        activation="relu",
        use_bn=False,
        dropout=0.0,
        regularizer=None,
        reg_coeff=0.0,
        lr=1e-3,
        batch_size=64,
        epochs=10,
        weight_decay=1e-4,
        scheduler="none",
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        early_stop_patience=5,
    ),

    "exp26_basic_wider": ExperimentConfig(
        hidden_sizes=[1024, 512],
        activation="relu",
        use_bn=False,
        dropout=0.0,
        regularizer=None,
        reg_coeff=0.0,
        lr=1e-3,
        batch_size=64,
        epochs=10,
        weight_decay=1e-4,
        scheduler="none",
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        early_stop_patience=5,
    ),

    "exp27_basic_l2": ExperimentConfig(
        hidden_sizes=[512, 256],
        activation="relu",
        use_bn=False,
        dropout=0.0,
        regularizer="l2",
        reg_coeff=1e-4,
        lr=1e-3,
        batch_size=64,
        epochs=10,
        weight_decay=0.0,
        scheduler="none",
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        early_stop_patience=5,
    ),

    "exp28_basic_gelu": ExperimentConfig(
        hidden_sizes=[512, 256],
        activation="gelu",
        use_bn=False,
        dropout=0.0,
        regularizer=None,
        reg_coeff=0.0,
        lr=1e-3,
        batch_size=64,
        epochs=10,
        weight_decay=0.0,
        scheduler="none",
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        early_stop_patience=5,
    ),
}


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

def get_params() -> tuple:
    """Parse CLI arguments and return (Args, GlobalConfig).

    Returns:
        Tuple of (Args, GlobalConfig). When args.run_all is False,
        use EXPERIMENTS[args.experiment] to get the ExperimentConfig.
    """
    parser = argparse.ArgumentParser(description="MLP on MNIST")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--experiment",
        choices=list(EXPERIMENTS.keys()),
        help="Name of a single experiment config to run.",
    )
    group.add_argument(
        "--run_all",
        action="store_true",
        help="Run all experiments sequentially.",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test", "both"],
        required=True,
        help="Training mode: 'train', 'test', or 'both'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Device: 'cpu' or 'cuda'.",
    )
    parsed = parser.parse_args()

    args       = Args(experiment=parsed.experiment, mode=parsed.mode,
                      device=parsed.device, run_all=parsed.run_all)
    global_cfg = GlobalConfig()

    return args, global_cfg


"""
Default config

import argparse
from dataclasses import dataclass
from typing import List


def get_params():
    parser = argparse.ArgumentParser(description="MLP on MNIST")
    parser.add_argument("--mode",      choices=["train", "test", "both"], default="both")
    parser.add_argument("--epochs",    type=int,   default=10)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--device",    type=str,   default="cpu")
    parser.add_argument("--batch_size",type=int,   default=64)
    args = parser.parse_args()

    return {
        # Data
        "data_dir":     "./data",
        "num_workers":  2,

        # Model
        "input_size":   784,        # 28x28
        "hidden_sizes": [512, 256, 128],
        "num_classes":  10,
        "dropout":      0.3,

        # Training
        "epochs":        args.epochs,
        "batch_size":    args.batch_size,
        "learning_rate": args.lr,
        "weight_decay":  1e-4,

        # Misc
        "seed":         42,
        "device":       args.device,
        "save_path":    "best_model.pth",
        "log_interval": 100,        # print every N batches

        # CLI
        "mode":         args.mode,
    }
"""