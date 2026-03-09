"""
visualize_results.py

Reads results/history.json and produces comparison plots for three experiment sets:

  Set 1 — Training framework comparison    : exp1, exp12–exp16
  Set 2 — Baseline architecture comparison : exp1–exp11, exp22, exp23_l1_high_coeff, exp24_l2_high_coeff
  Set 3 — Basic model architecture         : exp17–exp21, exp23_basic_deeper,
                                             exp24_basic_shallower, exp25–exp28

All plots are saved to results/.
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

HISTORY_FILE = os.path.join("results", "history.json")
RESULTS_DIR  = "results"

# ---------------------------------------------------------------------------
# Experiment set definitions
# ---------------------------------------------------------------------------

TRAINING_EXPS = [
    "exp1_baseline", "exp12_cosine_scheduler", "exp13_high_lr",
    "exp14_low_lr", "exp15_less_epoch", "exp16_more_epoch",
]

BASELINE_ARCH_EXPS = [
    "exp1_baseline", "exp2_gelu", "exp3_deep", "exp4_shallow",
    "exp5_wide", "exp6_narrow", "exp7_no_dropout", "exp8_high_dropout",
    "exp9_no_bn", "exp10_l1_reg", "exp11_l2_reg",
    "exp22_bn_after_activation", "exp23_l1_high_coeff", "exp24_l2_high_coeff",
]

BASIC_ARCH_EXPS = [
    "exp17_basic", "exp18_basic_dropout", "exp19_basic_scheduler",
    "exp20_basic_bn", "exp21_basic_l1", "exp23_basic_deeper",
    "exp24_basic_shallower", "exp25_basic_narrower", "exp26_basic_wider",
    "exp27_basic_l2", "exp28_basic_gelu", "exp29_basic_bn_after"
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_history() -> dict:
    """Load history.json and return a dict keyed by experiment name.

    Returns the most recent entry for each experiment name.
    """
    if not os.path.exists(HISTORY_FILE):
        raise FileNotFoundError(
            f"History file not found at '{HISTORY_FILE}'. "
            "Run experiments first with: python main.py --run_all --mode both"
        )
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)

    latest = {}
    for entry in history:
        latest[entry["experiment"]] = entry
    return latest


def _bar_chart(
    exp_names: list,
    values: list,
    title: str,
    ylabel: str,
    filename: str,
    ascending: bool = False,
    color: str = "steelblue",
) -> None:
    """Save a bar chart comparing experiments, sorted best to worst.

    Args:
        exp_names: Experiment name keys.
        values: Corresponding metric values.
        title: Chart title.
        ylabel: Y-axis label.
        filename: Output filename under results/.
        ascending: True = sort low→high (best for loss); False = high→low (best for accuracy).
        color: Bar color.
    """
    paired = sorted(zip(values, exp_names), reverse=not ascending)
    values = [v for v, _ in paired]
    names  = [n for _, n in paired]

    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
    bars = ax.bar(x, values, color=color, edgecolor="white", width=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.set_ylim(0, max(values) * 1.12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


def _heatmap(
    exp_names: list,
    per_class: dict,
    overall_acc: dict,
    title: str,
    filename: str,
) -> None:
    """Save a heatmap of per-class accuracy, rows sorted best to worst.

    Args:
        exp_names: Experiment name keys.
        per_class: Dict mapping exp_name → {digit_str: accuracy}.
        overall_acc: Dict mapping exp_name → overall accuracy (used for sorting).
        title: Chart title.
        filename: Output filename.
    """
    exp_names = sorted(exp_names, key=lambda e: overall_acc[e], reverse=True)

    num_classes = 10
    matrix = np.array([
        [per_class[e][str(d)] for d in range(num_classes)]
        for e in exp_names
    ])

    fig, ax = plt.subplots(figsize=(11, max(4, len(exp_names) * 0.65)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGn", vmin=0.9, vmax=1.0)

    ax.set_xticks(range(num_classes))
    ax.set_xticklabels([f"Digit {d}" for d in range(num_classes)], fontsize=8)
    ax.set_yticks(range(len(exp_names)))
    ax.set_yticklabels(exp_names, fontsize=8)
    ax.set_title(title, fontsize=12, fontweight="bold")

    for i in range(len(exp_names)):
        for j in range(num_classes):
            ax.text(j, i, f"{matrix[i, j]:.3f}",
                    ha="center", va="center", fontsize=7, color="black")

    fig.colorbar(im, ax=ax, label="Accuracy")
    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_set(
    history: dict,
    exp_list: list,
    set_label: str,
    acc_file: str,
    loss_file: str,
    heatmap_file: str,
    acc_color: str,
    loss_color: str,
) -> None:
    """Generate accuracy bar, loss bar, and per-class heatmap for one experiment set.

    Args:
        history: Full history dict keyed by experiment name.
        exp_list: Ordered list of experiment names for this set.
        set_label: Human-readable set name for plot titles.
        acc_file: Filename for accuracy bar chart.
        loss_file: Filename for loss bar chart.
        heatmap_file: Filename for per-class heatmap.
        acc_color: Bar color for accuracy chart.
        loss_color: Bar color for loss chart.
    """
    present = [e for e in exp_list if e in history]
    missing = [e for e in exp_list if e not in history]
    if missing:
        print(f"  [warning] Not yet run: {missing}")
    if not present:
        print(f"  [skip] No data available for this set.")
        return

    accuracies  = [history[e]["accuracy"]  for e in present]
    losses      = [history[e]["test_loss"] for e in present]
    per_class   = {e: history[e]["per_class_accuracy"] for e in present}
    overall_acc = {e: history[e]["accuracy"] for e in present}

    _bar_chart(
        present, accuracies,
        title     = f"{set_label} — Test Accuracy",
        ylabel    = "Test Accuracy",
        filename  = acc_file,
        ascending = False,
        color     = acc_color,
    )
    _bar_chart(
        present, losses,
        title     = f"{set_label} — Test Loss",
        ylabel    = "Test Loss",
        filename  = loss_file,
        ascending = True,
        color     = loss_color,
    )
    _heatmap(
        present, per_class, overall_acc,
        title    = f"{set_label} — Per-Class Accuracy",
        filename = heatmap_file,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    history = load_history()
    print(f"Loaded {len(history)} experiment(s) from {HISTORY_FILE}\n")

    print("--- Set 1: Training framework (exp1, exp12–exp16) ---")
    plot_set(
        history, TRAINING_EXPS,
        set_label    = "Training Framework Comparison",
        acc_file     = "training_accuracy.png",
        loss_file    = "training_loss.png",
        heatmap_file = "training_per_class_heatmap.png",
        acc_color    = "mediumseagreen",
        loss_color   = "darkorange",
    )

    print("\n--- Set 2: Baseline architecture (exp1–exp11, exp22, exp23_l1_high_coeff, exp24_l2_high_coeff) ---")
    plot_set(
        history, BASELINE_ARCH_EXPS,
        set_label    = "Baseline Architecture Comparison",
        acc_file     = "baseline_arch_accuracy.png",
        loss_file    = "baseline_arch_loss.png",
        heatmap_file = "baseline_arch_per_class_heatmap.png",
        acc_color    = "steelblue",
        loss_color   = "tomato",
    )

    print("\n--- Set 3: Basic model architecture (exp17–exp21, exp23_basic_deeper, exp24_basic_shallower, exp25–exp28) ---")
    plot_set(
        history, BASIC_ARCH_EXPS,
        set_label    = "Basic Model Architecture Comparison",
        acc_file     = "basic_arch_accuracy.png",
        loss_file    = "basic_arch_loss.png",
        heatmap_file = "basic_arch_per_class_heatmap.png",
        acc_color    = "mediumpurple",
        loss_color   = "slategray",
    )

    print("\nDone. All plots saved to results/")


if __name__ == "__main__":
    main()
