"""
visualize.py — run every experiment in this repo and produce the README figures.

It re-uses the `qfl` package as-is (data, models, Client, fedavg, accuracy) and
mirrors the settings of the existing scripts:

    federated  quantum / small-classical  -> runCompare.py  (3 clients, 2 epochs, lr 0.1)
    federated  64-feature MLP             -> run.py         (3 clients, 3 epochs, lr 0.01)
    centralized quantum                   -> quantumCentralized.py (20 epochs, lr 0.1)
    centralized small-classical           -> classicalSmall.py     (20 epochs, lr 0.1)
    centralized 64-feature MLP            -> centralized.py        (40 epochs, lr 0.01)

Usage
    python visualize.py                  # train everything, then plot
    python visualize.py --rounds 20      # fewer federated rounds (faster)
    python visualize.py --plot-only      # re-plot from results/metrics.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from qfl.data import (load_data, load_quantum_data,
                      split_with_missing_digits, include_digits)
from qfl.models import make_model, make_quantum_model, make_small_classical
from qfl.client import Client
from qfl.server import fedavg, accuracy

RESULTS = Path(__file__).parent / "results"
METRICS_FILE = RESULTS / "metrics.json"

MISSING_DIGITS = [[1, 3, 7], [2, 5, 8], [4, 6, 9]]
GROUPS = {"137": [1, 3, 7], "258": [2, 5, 8], "469": [4, 6, 9]}

# Each model's federated / centralized hyper-parameters (copied from the scripts).
MODELS = {
    "quantum": dict(
        label="Quantum VQC (8 qubits)", data="pca8",
        build=lambda: make_quantum_model(n_qubits=8, n_layers=2, n_classes=10),
        fl=dict(epochs=2, lr=0.1), central=dict(epochs=20, lr=0.1)),
    "small_classical": dict(
        label="Small classical (8→7→10)", data="pca8",
        build=lambda: make_small_classical(n_features=8, hidden=7, n_classes=10),
        fl=dict(epochs=2, lr=0.1), central=dict(epochs=20, lr=0.1)),
    "mlp": dict(
        label="Classical MLP (64→32→10)", data="raw64",
        build=make_model,
        fl=dict(epochs=3, lr=0.01), central=dict(epochs=40, lr=0.01)),
}

# ---------- palette (validated categorical slots 1-3, light surface) ----------
COLORS = {"quantum": "#2a78d6", "small_classical": "#eb6834", "mlp": "#1baf7a"}
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SURFACE, BASELINE_GRAY = "#fcfcfb", "#c9c8c2"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f3f7fd", "#bcd6f5", "#6ba3e8", "#2a78d6", "#174a8c"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10,
    "axes.edgecolor": BASELINE_GRAY, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": False, "lines.linewidth": 2,
})


# ============================================================================
# Experiments
# ============================================================================
def get_data(kind):
    return load_quantum_data(n_qubits=8) if kind == "pca8" else load_data()


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def run_federated(name, rounds):
    cfg = MODELS[name]
    X_train, y_train, X_test, y_test = get_data(cfg["data"])
    parts = split_with_missing_digits(y_train.numpy(), MISSING_DIGITS)
    clients = [Client(X_train[idx], y_train[idx]) for idx in parts]
    tests = {g: include_digits(X_test, y_test, d) for g, d in GROUPS.items()}

    torch.manual_seed(0)
    global_model = cfg["build"]()
    hist = {"all": [], **{g: [] for g in GROUPS}}

    for rnd in range(1, rounds + 1):
        t0 = time.time()
        trained, sizes = [], []
        for c in clients:
            trained.append(c.train(global_model, **cfg["fl"]))
            sizes.append(len(c.y))
        global_model.load_state_dict(fedavg(trained, sizes))

        hist["all"].append(accuracy(global_model, X_test, y_test))
        for g, (Xg, yg) in tests.items():
            hist[g].append(accuracy(global_model, Xg, yg))
        print(f"  [{name}] round {rnd:3d}/{rounds} | all {hist['all'][-1]:.3f} | "
              + " | ".join(f"{g} {hist[g][-1]:.3f}" for g in GROUPS)
              + f"  ({time.time() - t0:.1f}s)")

    with torch.no_grad():
        preds = global_model(X_test).argmax(dim=1)
    conf = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_test.tolist(), preds.tolist()):
        conf[t, p] += 1
    return hist, conf.tolist(), n_params(global_model)


def run_centralized(name):
    cfg = MODELS[name]
    X_train, y_train, X_test, y_test = get_data(cfg["data"])
    torch.manual_seed(0)
    model = cfg["build"]()
    model = Client(X_train, y_train).train(model, **cfg["central"])
    acc = accuracy(model, X_test, y_test)
    print(f"  [{name}] centralized accuracy {acc:.3f}")
    return acc


def client_distribution():
    _, y_train, _, _ = load_data()
    parts = split_with_missing_digits(y_train.numpy(), MISSING_DIGITS)
    return [np.bincount(y_train.numpy()[idx], minlength=10).tolist() for idx in parts]


def run_all(rounds):
    metrics = {"rounds": rounds, "clients": client_distribution(), "models": {}}
    for name in MODELS:
        print(f"\n== {MODELS[name]['label']} ==")
        t0 = time.time()
        hist, conf, params = run_federated(name, rounds)
        central = run_centralized(name)
        metrics["models"][name] = dict(
            label=MODELS[name]["label"], params=params, federated=hist,
            confusion=conf, centralized=central,
            seconds=round(time.time() - t0, 1))
    RESULTS.mkdir(exist_ok=True)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2))
    print(f"\nsaved {METRICS_FILE}")
    return metrics


# ============================================================================
# Figures
# ============================================================================
def save(fig, name):
    path = RESULTS / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def spread_labels(ys, min_gap):
    """Nudge end-of-line label positions apart so they don't overlap."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for i in range(1, len(order)):
        a, b = order[i - 1], order[i]
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out


def plot_client_distribution(m):
    counts = np.array(m["clients"])
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.grid(False)
    ax.imshow(counts, cmap=SEQ_BLUE, aspect="auto", vmin=0)
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            v = counts[i, j]
            if v == 0:
                ax.text(j, i, "missing", ha="center", va="center",
                        fontsize=8, color=MUTED, style="italic")
            else:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                        color="white" if v > counts.max() * 0.6 else INK)
    ax.set_xticks(range(10), [str(d) for d in range(10)])
    ax.set_yticks(range(len(counts)),
                  [f"Client {k}  (n={c.sum()})" for k, c in enumerate(counts)])
    ax.set_xlabel("Digit class")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Non-IID split: each client never sees three digit classes",
                 loc="left")
    save(fig, "client_distribution.png")


def plot_convergence(m):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    rounds = np.arange(1, m["rounds"] + 1)
    ends = []
    for name, d in m["models"].items():
        ax.plot(rounds, d["federated"]["all"], color=COLORS[name], label=d["label"])
        ends.append(d["federated"]["all"][-1])
    label_y = spread_labels(ends, 0.045)
    for (name, d), ly in zip(m["models"].items(), label_y):
        y = d["federated"]["all"][-1]
        ax.plot([rounds[-1]], [y], "o", ms=6, color=COLORS[name],
                mec=SURFACE, mew=1.5, zorder=5)
        ax.annotate(f"{y:.1%}", (rounds[-1], ly), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9,
                    color=INK_2, annotation_clip=False)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(1, rounds[-1])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Federated round")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_ylabel("Test accuracy (all digits)")
    ax.set_title("FedAvg convergence — global model accuracy per round", loc="left")
    ax.legend(loc="lower right")
    save(fig, "fl_convergence.png")


def plot_group_accuracy(m):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    rounds = np.arange(1, m["rounds"] + 1)
    for k, (ax, (g, digits)) in enumerate(zip(axes, GROUPS.items())):
        for name, d in m["models"].items():
            ax.plot(rounds, d["federated"][g], color=COLORS[name], label=d["label"])
        ax.set_title(f"Digits {', '.join(map(str, digits))}\n"
                     f"(absent from client {k})", loc="left", fontsize=10)
        ax.set_xlabel("Federated round")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_xlim(1, rounds[-1])
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_ylabel("Test accuracy on group")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Per-group accuracy: knowledge transfer to classes a client never saw",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "group_accuracy.png")


def plot_final_comparison(m):
    names = list(m["models"])
    x = np.arange(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.grid(axis="x", visible=False)
    for i, name in enumerate(names):
        d = m["models"][name]
        c, f = d["centralized"], d["federated"]["all"][-1]
        ax.bar(x[i] - w / 2 - 0.01, c, w, color=BASELINE_GRAY, zorder=3)
        ax.bar(x[i] + w / 2 + 0.01, f, w, color=COLORS[name], zorder=3)
        for xx, v in ((x[i] - w / 2 - 0.01, c), (x[i] + w / 2 + 0.01, f)):
            ax.text(xx, v + 0.015, f"{v:.1%}", ha="center", va="bottom",
                    fontsize=9, color=INK)
    ax.set_xticks(x, [f"{m['models'][n]['label']}\n{m['models'][n]['params']:,} params"
                      for n in names])
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Test accuracy (all digits)")
    ax.tick_params(axis="x", length=0)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BASELINE_GRAY, label="Centralized (all data, one site)"),
                       Patch(color=INK_2, label="Federated (3 non-IID clients, final round)")],
              loc="upper left", ncol=2, bbox_to_anchor=(0, 1.08))
    ax.set_title("Centralized vs. federated accuracy", loc="left", pad=28)
    save(fig, "final_comparison.png")


def plot_confusion(m, name="quantum"):
    d = m["models"][name]
    conf = np.array(d["confusion"])
    norm = conf / conf.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.grid(False)
    im = ax.imshow(norm, cmap=SEQ_BLUE, vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            if conf[i, j]:
                ax.text(j, i, conf[i, j], ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Row-normalised rate", color=INK_2)
    cb.outline.set_visible(False)
    ax.set_title(f"Confusion matrix — federated {d['label']}", loc="left", fontsize=10)
    save(fig, f"confusion_{name}.png")


def print_summary(m):
    print("\n| Model | Params | Centralized | Federated (final) | Best federated |")
    print("|---|---:|---:|---:|---:|")
    for d in m["models"].values():
        fl = d["federated"]["all"]
        print(f"| {d['label']} | {d['params']:,} | {d['centralized']:.1%} "
              f"| {fl[-1]:.1%} | {max(fl):.1%} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=50, help="federated rounds (default 50)")
    ap.add_argument("--plot-only", action="store_true",
                    help="skip training and plot from results/metrics.json")
    args = ap.parse_args()

    m = json.loads(METRICS_FILE.read_text()) if args.plot_only else run_all(args.rounds)
    RESULTS.mkdir(exist_ok=True)
    plot_client_distribution(m)
    plot_convergence(m)
    plot_group_accuracy(m)
    plot_final_comparison(m)
    plot_confusion(m, "quantum")
    print_summary(m)


if __name__ == "__main__":
    main()
