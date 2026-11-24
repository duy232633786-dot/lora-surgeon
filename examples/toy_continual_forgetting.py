#!/usr/bin/env python3
"""Reproducible toy experiment: sequential LoRA forgetting and weight-space de-forgetting.

Everything runs on CPU in a few seconds with only numpy + lora_surgeon. No model download.

Setup (per adapted module, 8 modules = 4 layers x {attn_q, ffn_down}):
  * A frozen "world" teacher  W* = D_tool + D_r1 + D_r2  (low-rank deltas on top of a base model).
  * Task TOOL   : inputs from subspace S_T, target W* x.
  * Task REASON : inputs from subspace S_R (partially overlapping S_T), target W* x.
  * Gen-1 : LoRA trained on TOOL only.                                  (clean tool-use baseline)
  * Gen-2 : warm-start Gen-1, trained on REASON-level-1 data only
            (teacher without D_r2) at a high learning rate.              (learns reasoning, forgets tools)
  * Gen-3 : warm-start Gen-2, trained on full REASON data (optional TOOL replay, default 0%)
            at a lower learning rate.                                    (best reasoning, tools still broken)
  Then lora_surgeon heals Gen-3 with `linear` and `orthogonal` de-forgetting and every adapter is scored
  on held-out TOOL and REASON inputs (relative squared error, lower is better).

The three generations are also written to ./toy_out/*.gguf so you can run the CLI on them yourself:
  lora-surgeon diff toy_out/gen1.gguf toy_out/gen2.gguf toy_out/gen3.gguf
  lora-surgeon deforget --v1 toy_out/gen1.gguf --v2 toy_out/gen2.gguf --v3 toy_out/gen3.gguf -o toy_out/healed.gguf

This is a controlled sandbox, not evidence about any real LLM: forgetting here comes only from the
shared low-rank factors of a LoRA being pushed around by sequential training.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import lora_surgeon as ls  # noqa: E402

DEFAULT_RANK = 16
DEFAULT_REPLAY = 0.0
MODULE_SHAPES = {
    f"blk.{i}.{name}": shape
    for i in range(4)
    for name, shape in (("attn_q.weight", (64, 64)), ("ffn_down.weight", (64, 128)))
}


class Adam:
    def __init__(self, params: list[np.ndarray], lr: float):
        self.params, self.lr, self.t = params, lr, 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads: list[np.ndarray]) -> None:
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            p -= self.lr * (m / (1 - b1**self.t)) / (np.sqrt(v / (1 - b2**self.t)) + eps)


def make_world(rng: np.random.Generator, d_out: int, d_in: int) -> dict:
    q, _ = np.linalg.qr(rng.standard_normal((d_in, d_in)))
    s_t = q[:, 0:24]          # TOOL input subspace
    s_r = q[:, 14:38]         # REASON input subspace (10 dims shared with TOOL)

    def low_rank_on(sub: np.ndarray, k: int, scale: float) -> np.ndarray:
        u = rng.standard_normal((d_out, k)) / np.sqrt(d_out)
        v = rng.standard_normal((k, sub.shape[1])) / np.sqrt(sub.shape[1])
        return scale * (u @ v @ sub.T)

    d_tool = low_rank_on(s_t, 5, 3.0)
    d_r1 = low_rank_on(s_r, 4, 3.0)
    d_r2 = low_rank_on(s_r, 3, 3.0)
    return {"s_t": s_t, "s_r": s_r, "d_tool": d_tool, "d_r1": d_r1, "d_r2": d_r2}


def sample(rng: np.random.Generator, sub: np.ndarray, n: int) -> np.ndarray:
    return (sub @ rng.standard_normal((sub.shape[1], n))).astype(np.float32)  # (d_in, n)


def train(b, a, batches: list[tuple[np.ndarray, np.ndarray, float]], lr: float, steps: int):
    """Full-batch Adam on weighted MSE of (B A) x vs y. Mutates and returns (b, a)."""
    opt = Adam([b, a], lr)
    for _ in range(steps):
        gb = np.zeros_like(b)
        ga = np.zeros_like(a)
        for x, y, w in batches:
            z = a @ x
            err = (b @ z - y) * (2.0 * w / x.shape[1])
            gb += err @ z.T
            ga += b.T @ err @ x.T
        opt.step([gb, ga])
    return b, a


def rel_err(dw: np.ndarray, target: np.ndarray, x: np.ndarray) -> float:
    return float(np.sum((dw @ x - target @ x) ** 2) / np.sum((target @ x) ** 2))


def build_generations(seed: int, rank: int, replay: float, n_train: int = 256):
    rng = np.random.default_rng(seed)
    evals = {}
    gens = {1: {}, 2: {}, 3: {}}
    for key, (d_out, d_in) in MODULE_SHAPES.items():
        w = make_world(rng, d_out, d_in)
        full = w["d_tool"] + w["d_r1"] + w["d_r2"]
        lvl1 = w["d_tool"] + w["d_r1"]
        x_t, x_r = sample(rng, w["s_t"], n_train), sample(rng, w["s_r"], n_train)
        evals[key] = (full, sample(rng, w["s_t"], 1024), sample(rng, w["s_r"], 1024))

        b = np.zeros((d_out, rank), dtype=np.float32)
        a = (rng.standard_normal((rank, d_in)) / np.sqrt(d_in)).astype(np.float32)
        # Gen-1: clean tool-use LoRA
        b, a = train(b, a, [(x_t, full @ x_t, 1.0)], lr=1e-2, steps=600)
        gens[1][key] = (b.copy(), a.copy())
        # Gen-2: high-LR reasoning-only update, no tool data
        b, a = train(b, a, [(x_r, lvl1 @ x_r, 1.0)], lr=3e-2, steps=150)
        gens[2][key] = (b.copy(), a.copy())
        # Gen-3: continual SFT from Gen-2 on full reasoning (+ optional tool replay), lower LR
        b, a = train(b, a, [(x_r, full @ x_r, 1.0 - replay), (x_t, full @ x_t, replay)], lr=5e-3, steps=400)
        gens[3][key] = (b.copy(), a.copy())

    adapters = {
        g: ls.LoRAAdapter(modules=mods, alpha=float(rank), arch="llama", base_model="toy-world", name=f"gen{g}")
        for g, mods in gens.items()
    }
    return adapters, evals


def score(adapter: ls.LoRAAdapter, evals: dict) -> tuple[float, float]:
    tool, reason = [], []
    for key, (full, x_t, x_r) in evals.items():
        b, a = adapter.modules[key]
        dw = adapter.scale * (b.astype(np.float64) @ a.astype(np.float64))
        tool.append(rel_err(dw, full, x_t))
        reason.append(rel_err(dw, full, x_r))
    return float(np.mean(tool)), float(np.mean(reason))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.2, 0.4, 0.6])
    ap.add_argument("--protect-energy", type=float, default=0.85)
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parent / "toy_out"))
    ap.add_argument("--rank", type=int, default=DEFAULT_RANK)
    ap.add_argument("--replay", type=float, default=DEFAULT_REPLAY, help="Share of TOOL data replayed in Gen-3")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    t0 = time.time()
    rows: dict[str, list[tuple[float, float]]] = {}
    for i, seed in enumerate(args.seeds):
        gens, evals = build_generations(seed, args.rank, args.replay)
        if i == 0:
            out = pathlib.Path(args.out)
            for g, ad in gens.items():
                ls.save_gguf(ad, out / f"gen{g}.gguf")
        for g in (1, 2, 3):
            rows.setdefault(f"Gen-{g}", []).append(score(gens[g], evals))
        for mode in ("linear", "orthogonal"):
            for gamma in args.gammas:
                healed, _ = ls.run_deforget(
                    gens[1], gens[2], gens[3], mode=mode, gamma_attn=gamma, gamma_mlp=gamma,
                    protect_energy=args.protect_energy,
                )
                rows.setdefault(f"{mode} gamma={gamma:.1f}", []).append(score(healed, evals))

    g3_tool, g3_reason = np.mean(rows["Gen-3"], axis=0) * 100
    print(f"seeds={args.seeds} rank={args.rank} replay={args.replay} modules={len(MODULE_SHAPES)} "
          f"protect_energy={args.protect_energy}")
    print("Held-out relative squared error x100 (mean over seeds, lower is better)\n")
    print("| Adapter | TOOL error | REASON error | TOOL vs Gen-3 | REASON vs Gen-3 |")
    print("|---|---:|---:|---:|---:|")
    for name, vals in rows.items():
        tool, reason = np.mean(vals, axis=0) * 100
        print(f"| {name} | {tool:.2f} | {reason:.2f} | {tool - g3_tool:+.2f} | {reason - g3_reason:+.2f} |")
    print(f"\nDone in {time.time() - t0:.1f}s. Gen-1/2/3 adapters (seed {args.seeds[0]}) written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
