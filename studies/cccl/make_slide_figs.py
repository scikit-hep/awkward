# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Generate presentation figures for the lazy-fusion slides.

Uses the measured ``bench_lazy_fusion.py`` numbers:
  - CPU arm from ``bench.json`` (this repo);
  - GPU arm (della, A100) from the runs recorded in
    ``bench_lazy_fusion_results.md``.

Produces, into ``studies/cccl/figs/``:
  lazy_fusion_speedup.png      speedup vs chain depth (GPU + CPU)
  lazy_fusion_flat_time.png    the fusion signature: flat fused vs linear eager
  lazy_fusion_kernels.png      kernel launches: N eager ops -> 1 fused
  lazy_fusion_reduce.png       transform+reduce: folded op stays flat
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- palette (matches the Marp deck) --------------------------------------
GREEN = "#76B900"  # NVIDIA green
FAST = "#4c8c00"
SLOW = "#b23b2e"
INK = "#123f4d"
MUTE = "#9aa8ad"
GRID = "#dfe6e8"
BLUE = "#2a78d6"

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "axes.edgecolor": INK,
        "axes.linewidth": 1.2,
        "axes.titlecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
    }
)

# ---- measured data ---------------------------------------------------------
OPS = [2, 4, 8, 16, 32]  # element-wise ops (chain depth 1,2,4,8,16 x2)

# GPU (della, A100) — map-only chain `* 1.001 + 0.5`
GPU = {
    "200k lists": {
        "interp": [2.233, 4.292, 8.609, 17.160, 34.239],
        "fused": [0.191, 0.204, 0.229, 0.280, 0.378],
        "speedup": [11.72, 21.06, 37.63, 61.39, 90.68],
    },
    "2M lists": {
        "interp": [2.168, 4.273, 8.544, 17.005, 34.141],
        "fused": [0.189, 0.203, 0.229, 0.280, 0.379],
        "speedup": [11.48, 21.02, 37.37, 60.83, 89.98],
    },
}

# CPU (from bench_lazy_fusion_results.md)
CPU = {
    "200k lists": {"speedup": [5.46, 6.11, 7.04, 7.30, 8.40]},
    "2M lists": {"speedup": [1.65, 2.09, 2.52, 2.72, 2.88]},
}

# GPU transform+reduce (2M): A = separate map+reduce, B = folded op into reduce
REDUCE = {
    "separate + reduce": [0.157, 0.205, 0.312, 0.510, 0.915],
    "fused (folded op -> reduce)": [0.177, 0.178, 0.177, 0.181, 0.176],
}

FIGDIR = pathlib.Path(__file__).parent / "figs"
FIGDIR.mkdir(exist_ok=True)


def _finish(ax, fname, title, xlabel, ylabel, legend_loc="best"):
    ax.set_title(title, pad=12, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(OPS)
    ax.set_xticklabels([str(o) for o in OPS])
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if legend_loc:
        ax.legend(loc=legend_loc, frameon=False, fontsize=13)
    out = FIGDIR / fname
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print("wrote", out)


def fig_speedup():
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    # GPU 200k and 2M coincide within ~1% -> one line makes the point that the
    # GPU win is size-independent (launch + on-chip reuse, not dispatch).
    gpu = [(a + b) / 2 for a, b in
           zip(GPU["200k lists"]["speedup"], GPU["2M lists"]["speedup"])]
    ax.plot(OPS, gpu, "-o", color=GREEN, lw=3.4, ms=9,
            label="GPU A100  (200k & 2M coincide)")
    ax.plot(OPS, CPU["200k lists"]["speedup"], "--s", color=BLUE, lw=2.4, ms=7,
            label="CPU · 200k lists  (dispatch-bound)")
    ax.plot(OPS, CPU["2M lists"]["speedup"], "--s", color=MUTE, lw=2.4, ms=7,
            label="CPU · 2M lists  (bandwidth-bound)")
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100])
    ax.set_yticklabels(["1x", "2x", "5x", "10x", "20x", "50x", "100x"])
    ax.annotate("~90x", xy=(31.3, 88), xytext=(24, 44),
                color=GREEN, weight="bold", fontsize=17,
                arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 2})
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1,
              frameon=False, fontsize=12)
    _finish(ax, "lazy_fusion_speedup.png",
            "Fusion speedup vs chain length  (compute(fuse=True) vs eager)",
            "element-wise ops in the chain", "speedup  (log scale)",
            legend_loc=None)


def fig_flat_time():
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    interp = GPU["2M lists"]["interp"]
    fused = GPU["2M lists"]["fused"]
    ax.plot(OPS, interp, "-o", color=SLOW, lw=3, ms=8,
            label="eager interpreter  (1 launch / op)")
    ax.plot(OPS, fused, "-o", color=GREEN, lw=3, ms=8,
            label="fused  (1 kernel, total)")
    ax.fill_between(OPS, fused, interp, color=GREEN, alpha=0.06)
    ax.annotate("32 ops -> 1 kernel\n0.38 ms  (flat)",
                xy=(32, 0.379), xytext=(15, 8),
                color=GREEN, weight="bold", fontsize=14,
                arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 2})
    ax.annotate("34 ms", xy=(32, 34.141), xytext=(24.5, 26),
                color=SLOW, weight="bold", fontsize=14)
    _finish(ax, "lazy_fusion_flat_time.png",
            "The fusion signature  (GPU A100, 2M lists)",
            "element-wise ops in the chain", "time per call  (ms)",
            legend_loc="upper left")


def fig_kernels():
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = range(len(OPS))
    w = 0.38
    ax.bar([i - w / 2 for i in x], OPS, width=w, color=SLOW, label="eager")
    ax.bar([i + w / 2 for i in x], [1] * len(OPS), width=w, color=GREEN,
           label="fused")
    for i, o in enumerate(OPS):
        ax.text(i - w / 2, o + 0.6, str(o), ha="center", color=SLOW,
                fontsize=12, weight="bold")
        ax.text(i + w / 2, 1 + 0.6, "1", ha="center", color=FAST,
                fontsize=12, weight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(o) for o in OPS])
    ax.set_title("GPU kernel launches: N element-wise ops become 1",
                 pad=12, weight="bold")
    ax.set_xlabel("element-wise ops in the chain")
    ax.set_ylabel("CUDA transform-kernel launches")
    ax.grid(True, axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=13)
    plt.tight_layout()
    out = FIGDIR / "lazy_fusion_kernels.png"
    plt.savefig(out)
    plt.close()
    print("wrote", out)


def fig_reduce():
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(OPS, REDUCE["separate + reduce"], "-o", color=SLOW, lw=3, ms=8,
            label="separate map+reduce")
    ax.plot(OPS, REDUCE["fused (folded op -> reduce)"], "-o", color=GREEN,
            lw=3, ms=8, label="fused: folded map into reduce")
    ax.annotate("map fused into the reduce\n0.18 ms  (flat)",
                xy=(32, 0.176), xytext=(12, 0.55),
                color=GREEN, weight="bold", fontsize=13,
                arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 2})
    _finish(ax, "lazy_fusion_reduce.png",
            "Transform + sum reduction fused into one kernel  (GPU, 2M)",
            "element-wise ops before the reduction", "time per call  (ms)",
            legend_loc="upper left")


if __name__ == "__main__":
    fig_speedup()
    fig_flat_time()
    fig_kernels()
    fig_reduce()
    print("done ->", FIGDIR)
