import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_log(path):
    series = defaultdict(lambda: ([], []))
    with open(path) as f:
        for row in csv.DictReader(f):
            steps, split, value = int(row["step"]), row["split"], float(row["value"])
            series[split][0].append(steps)
            series[split][1].append(value)
    return series


# GPT-2 124M baseline numbers (Radford et al. 2019 / Karpathy's llm.c), for comparison
# against our pretraining run.
loss_baseline = {
    "124M": 3.2924,
}
hella2_baseline = {  # HellaSwag for GPT-2
    "124M": 0.294463,
}
hella3_baseline = {  # HellaSwag for GPT-3
    "124M": 0.337,
}


def annotate_last(ax, steps, values, color=None):
    if not steps:
        return
    x, y = steps[-1], values[-1]
    ax.annotate(f"{y:.4f}", xy=(x, y), xytext=(5, 0), textcoords="offset points",
                va="center", fontsize=8, color=color)


def plot_pretraining_or_sft(series, plot_path, is_pretraining):
    has_hellaswag = "hellaswag" in series  # only pretraining logs a hellaswag split

    fig, axes = plt.subplots(2 if has_hellaswag else 1, 1, figsize=(10, 8 if has_hellaswag else 4), sharex=True)
    loss_ax = axes[0] if has_hellaswag else axes

    for split in ("train", "val"):
        if split in series:
            steps, values = series[split]
            line, = loss_ax.plot(steps, values, label=split)
            if split == "val":
                annotate_last(loss_ax, steps, values, color=line.get_color())
    if is_pretraining:
        loss_ax.axhline(loss_baseline["124M"], color="tab:red", linestyle="--",
                         label="GPT-2 124M baseline")
    loss_ax.set_xlabel("step")
    loss_ax.set_ylabel("loss")
    loss_ax.set_title("Loss")
    loss_ax.legend()

    if has_hellaswag:
        acc_ax = axes[1]
        steps, values = series["hellaswag"]
        acc_ax.plot(steps, values, label="hellaswag", color="tab:green")
        annotate_last(acc_ax, steps, values, color="tab:green")
        if is_pretraining:
            acc_ax.axhline(hella2_baseline["124M"], color="tab:red", linestyle="--",
                            label="GPT-2 124M baseline")
            acc_ax.axhline(hella3_baseline["124M"], color="tab:purple", linestyle="--",
                            label="GPT-3 124M baseline")
        acc_ax.set_xlabel("step")
        acc_ax.set_ylabel("accuracy")
        acc_ax.set_title("HellaSwag accuracy")
        acc_ax.legend()

    fig.tight_layout()
    fig.savefig(plot_path)


def plot_rlvr(series, plot_path):
    # rlvr/finetune.py logs 'train_loss' (policy gradient loss), 'train_reward'
    # (mean rollout reward per step), and 'gsm8k_accuracy' (periodic pass@1 eval)
    # - a different shape of data than pretraining/sft's train/val/hellaswag, so
    # it gets its own layout instead of being forced into the same one.
    fig, (loss_ax, score_ax) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    steps, values = series["train_loss"]
    loss_ax.plot(steps, values, label="train_loss", color="tab:blue")
    loss_ax.set_xlabel("step")
    loss_ax.set_ylabel("loss")
    loss_ax.set_title("Policy gradient loss")
    loss_ax.legend()

    if "train_reward" in series:
        steps, values = series["train_reward"]
        score_ax.plot(steps, values, label="train_reward (rollout avg)", color="tab:orange")
    if "gsm8k_accuracy" in series:
        steps, values = series["gsm8k_accuracy"]
        score_ax.plot(steps, values, label="gsm8k_accuracy (eval)", color="tab:green")
    score_ax.set_xlabel("step")
    score_ax.set_ylabel("score (0-1)")
    score_ax.set_title("Reward / GSM8K accuracy")
    score_ax.legend()

    fig.tight_layout()
    fig.savefig(plot_path)


def main():
    parser = argparse.ArgumentParser(description="Plot a training run's log.csv")
    parser.add_argument("run", choices=["pretraining", "sft", "rlvr"], help="which run's log to plot")
    args = parser.parse_args()

    run_dir = os.path.join(os.path.dirname(__file__), args.run)
    log_path = os.path.join(run_dir, "log.csv")
    plot_path = os.path.join(run_dir, "plot.png")
    series = load_log(log_path)

    if args.run == "rlvr":
        plot_rlvr(series, plot_path)
    else:
        plot_pretraining_or_sft(series, plot_path, is_pretraining=args.run == "pretraining")

    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
