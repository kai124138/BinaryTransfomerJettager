#!/usr/bin/env python3
"""Regenerate the README figures from the versioned result tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

PRECISIONS = ("FP32", "W8A8", "W1A8", "W1A6", "W1A4")
COLORS = dict(zip(PRECISIONS, ("#243746", "#0072B2", "#009E73", "#D55E00", "#CC79A7")))
LABELS = {
    "tensor_quantization": "Tensor-wise baseline",
    "channel_quantization": "Channel-wise quantization",
    "reduced_feedforward": "Reduced feed-forward width",
    "attention_probability_8bit": "8-bit attention probabilities",
    "gradual_budget": "Gradual budget schedule",
    "fixed_width_recovery": "Fixed-width recovery",
    "knowledge_distillation": "Knowledge distillation",
}


def save(fig, directory, name):
    for extension in ("png", "svg"):
        path = directory / f"{name}.{extension}"
        metadata = {"Date": None} if extension == "svg" else {}
        fig.savefig(path, dpi=180, bbox_inches="tight", metadata=metadata)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def build(root: Path):
    results, figures = root / "results", root / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "semibold", "axes.titlesize": 14,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "svg.hashsalt": "binary-transformer-jet-tagging"})
    auc = pd.read_csv(results / "pre_conference/auc_per_seed.csv")
    summary = auc.groupby(["precision", "n_constituents"]).macro_ovr_auc.agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(9.5, 5), layout="constrained")
    for precision in PRECISIONS:
        data = summary.loc[precision]
        ax.errorbar(data.index, data["mean"], yerr=data["std"], label=precision,
                    marker="o", capsize=4, color=COLORS[precision], linewidth=2)
    ax.set(xlabel="Number of jet constituents", ylabel="Held-out macro one-vs-rest AUC",
           title="Pre-conference: constituent count and precision", xticks=[8, 16, 32, 64])
    ax.grid(alpha=.18)
    ax.legend(ncol=5, loc="lower right", fontsize=9)
    fig.supxlabel("Three training seeds; error bars show sample standard deviation", fontsize=10)
    save(fig, figures, "pre_conference_auc")

    ebops = json.loads((results / "pre_conference/ebops.json").read_text())["models"]
    costs = {(v["n_part"], v["variant"].upper(), v["seed"]): v["ebops"] for v in ebops.values()}
    fig, ax = plt.subplots(figsize=(9.5, 5), layout="constrained")
    for precision in PRECISIONS[1:]:
        group = auc[auc.precision == precision]
        x, y, errors = [], [], []
        for n, data in group.groupby("n_constituents"):
            cost = np.mean([costs[(n, precision, int(seed))] for seed in data.seed])
            x.append(cost); y.append(data.macro_ovr_auc.mean()); errors.append(data.macro_ovr_auc.std())
            ax.annotate(f"N={n}", (cost, y[-1]), xytext=(4, 7), textcoords="offset points", fontsize=8)
        ax.errorbar(x, y, yerr=errors, color=COLORS[precision], marker="o", capsize=4,
                    label=precision, linewidth=1.6)
    ax.set(xscale="log", xlabel="HGQ2 effective bit operations per inference",
           ylabel="Held-out macro one-vs-rest AUC", title="Pre-conference: accuracy–computation trade-off")
    ax.grid(alpha=.18); ax.legend(ncol=4, fontsize=9)
    fig.supxlabel("Fixed precision; no enforced EBOP budget. FP32 has no HGQ2 quantized-cost point.", fontsize=10)
    save(fig, figures, "pre_conference_ebops")

    curves = pd.read_csv(results / "pre_conference/roc_curves.csv")
    names = {"g": "Gluon", "q": "Light quark", "W": "W boson", "Z": "Z boson", "t": "Top quark"}
    for n in (8, 16):
        fig, axes = plt.subplots(1, 5, figsize=(15, 3.5), sharex=True, sharey=True, layout="constrained")
        for ax, label in zip(axes, names):
            for precision in PRECISIONS:
                data = curves[(curves.n_constituents == n) & (curves.precision == precision) & (curves["class"] == label)]
                ax.plot(data.fpr, data.tpr, color=COLORS[precision], label=precision)
            ax.set(xscale="log", xlim=(1e-4, 1), ylim=(0, 1.02), title=names[label], xlabel="Background efficiency")
            ax.grid(alpha=.18)
        axes[0].set_ylabel("Signal efficiency")
        axes[-1].legend(fontsize=8, loc="lower right")
        fig.suptitle(f"Pre-conference: one-vs-rest ROC curves, N = {n}, seed 1", fontsize=15)
        save(fig, figures, f"pre_conference_roc_n{n}")

    metrics = json.loads((results / "post_conference/ablation_metrics.json").read_text())
    rows = [r for r in metrics["runs"] if "ebops" in r]
    fig, ax = plt.subplots(figsize=(10.5, 5.5), layout="constrained")
    offsets = [(8, 9), (8, -15), (8, 10), (-8, 12), (-8, -14), (-8, -18), (-8, 10)]
    for row, (dx, dy) in zip(rows, offsets):
        interim = row["status"] != "finished"
        color = "#D55E00" if interim else "#0072B2"
        ax.scatter(row["ebops"], row["selected_val_macro_auc"], color=color,
                   marker="D" if interim else "o", s=70, zorder=3)
        label = LABELS[row["arm"]] + (" (interim)" if interim else "")
        ax.annotate(label, (row["ebops"], row["selected_val_macro_auc"]),
                    xytext=(dx, dy), textcoords="offset points", fontsize=9,
                    ha="left" if dx > 0 else "right")
    ax.axvline(metrics["budget_ebops"], color="#555", linestyle="--", label="350,000-EBOP constraint")
    ax.set(xlim=(313000, 355000), ylim=(.838, .858), xlabel="HGQ2 effective bit operations per inference",
           ylabel="Validation macro one-vs-rest AUC", title="Post-conference: EBOP-constrained training")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=.18)
    fig.supxlabel("Single seed; all selected checkpoints satisfy the 350,000-EBOP constraint.", fontsize=9)
    save(fig, figures, "post_conference_tradeoff")

    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    rows = sorted(rows, key=lambda r: r["test_accuracy"])
    labels = [LABELS[r["arm"]] + (" (interim)" if r["status"] != "finished" else "") for r in rows]
    values = [100 * r["test_accuracy"] for r in rows]
    ax.barh(labels, values, color=["#D55E00" if r["status"] != "finished" else "#0072B2" for r in rows], height=.6)
    for y, value in enumerate(values): ax.text(value + .6, y, f"{value:.2f}%", va="center", fontsize=10)
    ax.set(xlim=(0, 66), xlabel="Held-out top-1 categorical accuracy (%)",
           title="Post-conference: classification accuracy on 260,000 jets")
    ax.grid(axis="x", alpha=.18); ax.set_axisbelow(True)
    fig.supxlabel("Same checkpoints as the validation-AUC comparison; selection does not use held-out accuracy.", fontsize=9)
    save(fig, figures, "post_conference_accuracy")

    pilot = json.loads((results / "post_conference/budget_pilot_checkpoint_measurements.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    x = np.arange(4, 10)
    for idx, (key, label) in enumerate((("unconstrained_control", "Unconstrained control"),
                                      ("budget_75_percent", "75% budget (met)"),
                                      ("budget_25_percent", "25% budget (not met)"))):
        data = pilot[key]
        axes[0].bar(x + (idx - 1) * .24, [data["width_histogram"].get(str(b), 0) for b in x],
                    width=.24, label=label)
        axes[1].bar(idx, data["measured_ebops"] / 1e6, width=.6)
    axes[0].set(xlabel="Learned activation width (bits)", ylabel="Number of quantizers", xticks=x,
                title="Activation-width distributions")
    axes[0].legend(fontsize=8)
    axes[1].set(xticks=[0, 1, 2], xticklabels=["Control", "75% budget", "25% budget"],
                ylabel="Measured EBOPs (millions)", title="Selected checkpoint computation")
    for ax in axes: ax.grid(axis="y", alpha=.18); ax.set_axisbelow(True)
    fig.suptitle("Post-conference: initial budget-control study", fontsize=15)
    save(fig, figures, "budget_pilot_widths")

    hardware = json.loads((results / "hardware/hls_synthesis.json").read_text())["measurements"]
    selected = [r for r in hardware if r["configuration"] == "w1a8-s3-pre_conference_n8-softmax4_integer0"][0]
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    resources = ["LUT", "FF", "DSP", "BRAM_18K"]
    fractions = [100 * selected[k] / selected["avail"][k] for k in resources]
    ax.bar(resources, fractions, color=["#D55E00" if v > 100 else "#0072B2" for v in fractions])
    for x, y, k in zip(range(4), fractions, resources):
        ax.text(x, y + 5, f"{selected[k]:,}\n({y:.1f}%)", ha="center", fontsize=10)
    ax.axhline(100, color="#555", linestyle="--", label="Device capacity")
    ax.set(ylim=(0, 260), ylabel="HLS resource estimate / device capacity (%)",
           title="Hardware characterization: binary N = 8, 4-bit softmax")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=.18); ax.set_axisbelow(True)
    fig.supxlabel("VU13P target; HLS estimates precede logic optimization and place-and-route.", fontsize=9)
    save(fig, figures, "hardware_resources")
    print(f"Generated 8 figures (PNG and SVG) in {figures}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    build(parser.parse_args().root)
