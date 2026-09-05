from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path(__file__).with_name("preliminary_results_summary.png")


def main() -> None:
    colors = ["#7a7a7a", "#2b6cb0", "#c05621"]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))

    model_names = ["Free", "First-order", "Correctable"]
    one_dof_nmse = np.asarray([0.025113685, 0.026357008, 0.000187506])
    x = np.arange(len(model_names))
    axes[0].bar(x, one_dof_nmse, color=colors, width=0.68)
    axes[0].set_xticks(x, model_names)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Prediction NMSE (log scale)")
    axes[0].set_title("(a) 1-DoF: unseen negative actions")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].text(
        2,
        one_dof_nmse[2] * 1.45,
        "134x lower\nthan Free",
        ha="center",
        va="bottom",
        fontsize=9,
        color=colors[2],
    )

    categories = ["Negative\naxis", "Joint\n(+,+)", "Joint\n(-,-)", "Mixed\n(+,-)"]
    axis_only = np.asarray([0.002891395, 0.001953203, 0.006313373, 0.005759703])
    ten_percent = np.asarray([0.003378692, 0.000102445, 0.002981606, 0.001719991])
    x = np.arange(len(categories))
    width = 0.36
    bars_axis = axes[1].bar(
        x - width / 2,
        axis_only,
        width,
        label="Positive single-joint actions only",
        color="#718096",
    )
    bars_joint = axes[1].bar(
        x + width / 2,
        ten_percent,
        width,
        label="+ 10% positive joint actions",
        color="#c05621",
    )
    axes[1].set_xticks(x, categories)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Prediction NMSE (log scale)")
    axes[1].set_title("(b) 2-DoF: same correctable model")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].annotate(
        "19.1x lower",
        xy=(1 + width / 2, ten_percent[1]),
        xytext=(1.52, 0.00027),
        arrowprops={"arrowstyle": "->", "color": "#9c4221"},
        fontsize=9,
        color="#9c4221",
        ha="center",
    )

    fig.legend(
        [bars_axis, bars_joint],
        ["Positive single-joint actions only", "+ 10% positive joint actions"],
        loc="lower center",
        bbox_to_anchor=(0.72, -0.01),
        frameon=False,
        fontsize=9,
        ncol=2,
    )
    fig.suptitle("Preliminary evidence for local action-effect structure", y=1.02)
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

