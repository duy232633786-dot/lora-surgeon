#!/usr/bin/env python3
"""Regenerate the README figures (light + dark SVG) into docs/figures/.

    pip install matplotlib
    python examples/make_figures.py

`tradeoff` is computed live from examples/toy_continual_forgetting.py (5 seeds, about 20 s).
`generations` and `evalloss` plot the production numbers already tabulated in README.md.
`projection` is a schematic of the geometry, not data.
"""
from __future__ import annotations

import importlib.util
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"

# Validated categorical slots 1-2 (blue, orange) + neutral ink, stepped per mode. Surfaces match GitHub.
THEMES = {
    "light": dict(surface="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", blue="#2a78d6", orange="#eb6834", gray="#898781", band="#cde2fb"),
    "dark": dict(surface="#0d1117", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", blue="#3987e5", orange="#d95926", gray="#898781", band="#104281"),
}


def style(t: dict) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "text.color": t["ink"], "axes.labelcolor": t["ink2"], "axes.edgecolor": t["axis"],
        "xtick.color": t["muted"], "ytick.color": t["muted"], "xtick.labelcolor": t["ink2"],
        "ytick.labelcolor": t["ink2"], "axes.grid": True, "grid.color": t["grid"], "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "semibold",
        "axes.titlesize": 12, "axes.titlelocation": "left", "legend.frameon": False,
        "svg.fonttype": "path",
    })


def save(fig, name: str, mode: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}-{mode}.svg", bbox_inches="tight", pad_inches=0.15, metadata={"Date": None})
    plt.close(fig)


# ───────────────────────── 1. Toy trade-off (live data) ─────────────────────────

def toy_curves(gammas: list[float], seeds=(0, 1, 2, 3, 4)) -> dict:
    spec = importlib.util.spec_from_file_location("toy", ROOT / "examples" / "toy_continual_forgetting.py")
    toy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(toy)
    acc: dict[str, list] = {}
    for seed in seeds:
        gens, evals = toy.build_generations(seed, toy.DEFAULT_RANK, toy.DEFAULT_REPLAY)
        for g in (1, 2, 3):
            acc.setdefault(f"gen{g}", []).append(toy.score(gens[g], evals))
        for mode in ("linear", "orthogonal"):
            for gm in gammas:
                healed, _ = toy.ls.run_deforget(gens[1], gens[2], gens[3], mode=mode, gamma_attn=gm, gamma_mlp=gm)
                acc.setdefault(f"{mode}:{gm}", []).append(toy.score(healed, evals))
    return {k: np.mean(v, axis=0) * 100 for k, v in acc.items()}


def fig_tradeoff(res: dict, gammas: list[float], t: dict, mode: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    g3 = res["gen3"]
    for m, color, label in (("linear", t["orange"], "Linear subtraction"), ("orthogonal", t["blue"], "Orthogonal projection")):
        pts = np.array([g3] + [res[f"{m}:{g}"] for g in gammas])
        ax.plot(pts[:, 1], pts[:, 0], color=color, lw=2, zorder=3, label=label,
                marker="o", ms=6, markevery=list(range(1, len(pts))), mec=t["surface"], mew=1.5)
    for gm in (0.2, 0.4, 0.6):
        x, y = res[f"orthogonal:{gm}"][1], res[f"orthogonal:{gm}"][0]
        ax.annotate(f"γ={gm:.1f}", (x, y), xytext=(2, -15), textcoords="offset points", color=t["muted"],
                    fontsize=9, ha="left")
    for key, name, off in (("gen3", "Gen-3", (8, 6)), ("gen2", "Gen-2", (8, 4)), ("gen1", "Gen-1", (-8, 8))):
        y, x = res[key]
        ax.plot(x, y, "o", ms=9, color=t["ink"], mec=t["surface"], mew=2, zorder=4)
        ax.annotate(name, (x, y), xytext=off, textcoords="offset points", color=t["ink"], fontsize=10.5,
                    fontweight="semibold", ha="right" if off[0] < 0 else "left")
    ax.set_xlabel("REASON error  (cost of healing →)")
    ax.set_ylabel("TOOL error  (forgetting ↑)")
    ax.set_xlim(-2, 58)
    ax.set_ylim(-1.5, 28)
    ax.text(0.04, 0.04, "↙ lower-left is better", transform=ax.transAxes, ha="left", color=t["muted"], fontsize=9.5)
    ax.set_title("Toy sandbox: healing Gen-3, orthogonal vs linear (5 seeds, held-out error ×100)")
    ax.legend(loc="upper right", fontsize=10, labelcolor=t["ink2"])
    save(fig, "tradeoff", mode)


# ───────────────────────── 2. Production generations (README table) ─────────────────────────

SUITES = [  # (title, denominator, [Gen-1, Gen-2, Gen-3, De-forget])
    ("A1 · Tool selection", 199, [188, 185, 187, 188]),
    ("A2 · Exact tool arguments", 199, [128, 118, 119, 123]),
    ("B1 · Complex spec: tool / refusal", 32, [32, 28, 30, 30]),
    ("B2 · Complex spec: nested args", 32, [20, 19, 18, 18]),
    ("C · Multi-step reasoning", 202, [36, 72, 120, 79]),
    ("D · Diagram VQA", 63, [None, 50, 53, 52]),
]
STAGES = ["Gen-1", "Gen-2", "Gen-3", "Linear\nde-forget"]


def fig_generations(t: dict, mode: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6))
    for ax, (title, den, vals) in zip(axes.flat, SUITES):
        pct = [None if v is None else 100 * v / den for v in vals]
        xs = [i for i in range(3) if pct[i] is not None]
        ax.plot(xs, [pct[i] for i in xs], color=t["blue"], lw=2, marker="o", ms=7, mec=t["surface"], mew=1.5, zorder=3)
        ax.plot([2, 3], [pct[2], pct[3]], color=t["gray"], lw=1.2, zorder=2)
        ax.plot(3, pct[3], "o", ms=9, color=t["orange"], mec=t["surface"], mew=1.5, zorder=4)
        for i in xs + [3]:
            ax.annotate(f"{pct[i]:.1f}%", (i, pct[i]), xytext=(0, 8), textcoords="offset points", ha="center",
                        fontsize=9, color=t["ink"] if i == 3 else t["ink2"])
        valid = [p for p in pct if p is not None]
        lo, hi = min(valid), max(valid)
        pad = max(4.0, (hi - lo) * 0.35)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(-0.4, 3.4)
        ax.set_xticks(range(4), STAGES, fontsize=9)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, integer=True))
        ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=10.5)
    fig.suptitle("Production agent, 496-case frozen benchmark: what each generation and the linear fix did",
                 x=0.01, ha="left", fontsize=12.5, fontweight="semibold", color=t["ink"])
    fig.text(0.01, -0.01, "Blue: sequential training. Orange: Gen-3 healed with ΔW₃ − 0.40·(ΔW₂ − ΔW₁). "
             "Gen-1 has no VQA score.", color=t["muted"], fontsize=9.5)
    fig.tight_layout()
    save(fig, "generations", mode)


# ───────────────────────── 3. eval_loss mirage (README table) ─────────────────────────

def fig_evalloss(t: dict, mode: str) -> None:
    loss_steps = [0, 100, 400, 600, 800, 1000, 1200]
    loss = [0.4600, 0.2722, 0.1218, 0.0858, 0.0785, 0.0760, 0.0717]
    acc_steps = [100, 400, 600, 800, 1000, 1200]
    acc = [68.8, 70.0, 71.3, 73.8, 68.8, 68.8]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.2, 5.2), sharex=True, gridspec_kw={"hspace": 0.35})
    a1.plot(loss_steps, loss, color=t["gray"], lw=2, marker="o", ms=6, mec=t["surface"], mew=1.5)
    a1.set_title("Validation eval_loss keeps falling…")
    a1.set_ylim(0, 0.5)
    a1.annotate("0.0785", (800, 0.0785), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9, color=t["ink2"])
    a1.annotate("0.0717  lowest loss", (1200, 0.0717), xytext=(-4, 12), textcoords="offset points", ha="right",
                fontsize=9, color=t["ink2"])
    a2.plot(acc_steps, acc, color=t["blue"], lw=2, marker="o", ms=6, mec=t["surface"], mew=1.5)
    a2.plot(800, 73.8, "o", ms=10, color=t["blue"], mec=t["surface"], mew=2, zorder=4)
    a2.set_title("…while held-out exact tool-argument accuracy peaks at step 800 and drops")
    a2.set_ylim(66, 76)
    a2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    a2.annotate("73.8%  true best", (800, 73.8), xytext=(10, 0), textcoords="offset points", va="center",
                fontsize=9.5, color=t["ink"], fontweight="semibold")
    a2.annotate("68.8%", (1200, 68.8), xytext=(0, -15), textcoords="offset points", ha="center", fontsize=9, color=t["ink2"])
    for ax in (a1, a2):
        ax.axvline(800, color=t["axis"], lw=1, zorder=1)
        ax.grid(axis="x", visible=False)
    a2.set_xlabel("Gen-3 training step")
    a2.set_xticks(loss_steps)
    save(fig, "evalloss", mode)


# ───────────────────────── 4. Projection schematic ─────────────────────────

def fig_projection(t: dict, mode: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.fill_between([-0.6, 5.6], -0.28, 0.28, color=t["band"], alpha=0.55, lw=0, zorder=0)
    ax.text(5.6, -0.5, "protected subspace: top singular directions of (Gen-3 − Gen-2)\n= where the new skills live",
            ha="right", va="top", fontsize=9.5, color=t["ink2"])
    f = np.array([3.2, 2.6])

    def arrow(start, vec, color, lw=2.2):
        ax.annotate("", xy=start + vec, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=16, shrinkA=0, shrinkB=0))

    o = np.array([0.0, 0.0])
    arrow(o, f, t["gray"])
    ax.text(*(f * 0.5 + np.array([-1.15, 0.35])), "F = ΔW₂ − ΔW₁\nGen-2 drift", fontsize=10, color=t["ink"], ha="center")
    arrow(o, np.array([f[0], 0.0]), t["ink2"], lw=1.6)
    ax.text(f[0] * 0.5, -0.12, "kept: drift Gen-3 builds on", fontsize=9.5, color=t["ink2"], ha="center", va="top")
    arrow(np.array([f[0], 0.0]), np.array([0.0, f[1]]), t["blue"], lw=2.6)
    ax.text(f[0] + 0.15, f[1] * 0.55, "F_perp  removed by\n--mode orthogonal", fontsize=10, color=t["ink"],
            fontweight="semibold", va="center")
    ax.plot([f[0] - 0.22, f[0] - 0.22, f[0]], [0, 0.22, 0.22], color=t["ink2"], lw=1)
    ax.text(-0.6, 3.55, "--mode linear subtracts γ·F (all of it). --mode orthogonal subtracts only γ·F_perp,",
            fontsize=10.5, color=t["ink"])
    ax.text(-0.6, 3.2, "the part of Gen-2's drift that Gen-3's new skills do not use.", fontsize=10.5, color=t["ink"])
    ax.set_xlim(-0.7, 5.7)
    ax.set_ylim(-1.2, 3.9)
    save(fig, "projection", mode)


def main() -> int:
    gammas = [round(0.1 * i, 1) for i in range(1, 7)]
    res = toy_curves(gammas)
    for mode, t in THEMES.items():
        style(t)
        fig_tradeoff(res, gammas, t, mode)
        fig_generations(t, mode)
        fig_evalloss(t, mode)
        fig_projection(t, mode)
    print("wrote", sorted(p.name for p in OUT.glob("*.svg")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
