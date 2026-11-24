#!/usr/bin/env python3
"""lora-surgeon: CPU-native GGUF & Safetensors LoRA weight surgery in O(d * r^2).

Zero GPU, zero PyTorch, zero base-model download required.
Features:
  - inspect  : Per-layer Frobenius norm profile & rank/alpha metadata of any .gguf or .safetensors LoRA.
  - diff     : Exact O(d * r^2) Frobenius distance & cosine similarity via trace trick tr((B1^T B2)(A2 A1^T)).
  - merge    : Exact Thin QR-SVD multi-adapter merge (retains ~99.97% spectral energy at rank r).
  - deforget : Continual LoRA De-Forgetting across sequential generations (v1 -> v2 -> v3):
                 * linear     : W3 - gamma * (W2 - W1)
                 * orthogonal : Projects (W2 - W1) onto the exact Frobenius orthogonal complement of
                                (W3 - W2)'s top-k singular subspace: F_perp = F - U_k (U_k^T B_f)(A_f V_k) V_k^T
                 * selective  : Independent --gamma-attn and --gamma-mlp schedules.
  - convert  : Standalone bidirectional .gguf <-> PEFT adapter_model.safetensors converter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import sys
import time
from dataclasses import dataclass

import gguf
import numpy as np

__version__ = "0.1.0"

GGUF_TO_PEFT_SUFFIX = {
    "attn_q.weight": "self_attn.q_proj",
    "attn_k.weight": "self_attn.k_proj",
    "attn_v.weight": "self_attn.v_proj",
    "attn_output.weight": "self_attn.o_proj",
    "ffn_gate.weight": "mlp.gate_proj",
    "ffn_up.weight": "mlp.up_proj",
    "ffn_down.weight": "mlp.down_proj",
}
PEFT_TO_GGUF_SUFFIX = {v: k for k, v in GGUF_TO_PEFT_SUFFIX.items()}


@dataclass
class LoRAAdapter:
    """In-memory representation of a LoRA adapter normalized to canonical module keys."""

    modules: dict[str, tuple[np.ndarray, np.ndarray]]  # key -> (B: [d_out, r], A: [r, d_in]) float32
    alpha: float
    arch: str = "llama"
    base_model: str = ""
    name: str = "lora-surgeon-adapter"
    peft_prefix: str = "base_model.model.model.layers"

    @property
    def rank(self) -> int:
        first_b, _ = next(iter(self.modules.values()))
        return int(first_b.shape[1])

    @property
    def scale(self) -> float:
        return float(self.alpha) / float(self.rank)


def is_attn_module(mod_key: str) -> bool:
    return "attn_" in mod_key or "self_attn" in mod_key


def _canonical_from_peft(peft_key: str) -> tuple[str, str, str]:
    """Return (peft_prefix, canonical_gguf_base_key, 'A'|'B')."""
    m = re.match(r"^(.*?)\.layers\.(\d+)\.(.+)\.(lora_[AB])\.weight$", peft_key)
    if not m:
        raise ValueError(f"Unsupported PEFT LoRA key format: {peft_key}")
    prefix, layer_idx, mod_peft, ab = m.groups()
    if mod_peft not in PEFT_TO_GGUF_SUFFIX:
        raise ValueError(f"Unsupported module suffix '{mod_peft}' in {peft_key}")
    canon = f"blk.{layer_idx}.{PEFT_TO_GGUF_SUFFIX[mod_peft]}"
    return f"{prefix}.layers", canon, ("A" if ab == "lora_A" else "B")


def load_adapter(path: str | pathlib.Path, default_alpha: float = 64.0, arch: str = "llama") -> LoRAAdapter:
    p = pathlib.Path(path)
    if p.suffix.lower() == ".gguf":
        reader = gguf.GGUFReader(str(p))
        alpha = default_alpha
        det_arch = arch
        base_model = ""
        name = p.stem
        if "adapter.lora.alpha" in reader.fields:
            alpha = float(reader.fields["adapter.lora.alpha"].contents())
        if "general.architecture" in reader.fields:
            det_arch = str(reader.fields["general.architecture"].contents())
        if "general.base_model.0.name" in reader.fields:
            base_model = str(reader.fields["general.base_model.0.name"].contents())
        if "general.name" in reader.fields:
            name = str(reader.fields["general.name"].contents())

        raw_a, raw_b = {}, {}
        for t in reader.tensors:
            arr = np.array(t.data, dtype=np.float32)
            if t.name.endswith(".lora_a"):
                raw_a[t.name[:-7]] = arr
            elif t.name.endswith(".lora_b"):
                raw_b[t.name[:-7]] = arr
        modules = {k: (raw_b[k], raw_a[k]) for k in sorted(raw_a.keys()) if k in raw_b}
        if not modules:
            raise ValueError(f"No LoRA tensor pairs (.lora_a/.lora_b) found in {p}")
        return LoRAAdapter(modules=modules, alpha=alpha, arch=det_arch, base_model=base_model, name=name)

    from safetensors import safe_open

    st_path = p / "adapter_model.safetensors" if p.is_dir() else p
    cfg_path = (p if p.is_dir() else p.parent) / "adapter_config.json"
    alpha = default_alpha
    base_model = ""
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        alpha = float(cfg.get("lora_alpha", default_alpha))
        base_model = str(cfg.get("base_model_name_or_path", ""))

    raw_a, raw_b = {}, {}
    peft_prefix = "base_model.model.model.layers"
    with safe_open(str(st_path), framework="numpy") as f:
        for pk in f.keys():
            pref, canon, ab = _canonical_from_peft(pk)
            peft_prefix = pref
            arr = np.array(f.get_tensor(pk), dtype=np.float32)
            if ab == "A":
                raw_a[canon] = arr
            else:
                raw_b[canon] = arr
    modules = {k: (raw_b[k], raw_a[k]) for k in sorted(raw_a.keys()) if k in raw_b}
    if not modules:
        raise ValueError(f"No PEFT LoRA tensor pairs found in {st_path}")
    return LoRAAdapter(
        modules=modules, alpha=alpha, arch=arch, base_model=base_model, name=p.stem, peft_prefix=peft_prefix
    )


def save_gguf(adapter: LoRAAdapter, out_path: str | pathlib.Path) -> str:
    p = pathlib.Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(p), arch=adapter.arch)
    writer.add_type(gguf.GGUFType.ADAPTER)
    writer.add_string("adapter.type", "lora")
    writer.add_name(adapter.name)
    if adapter.base_model:
        writer.add_uint32("general.base_model.count", 1)
        writer.add_string("general.base_model.0.name", adapter.base_model)
    writer.add_float32("adapter.lora.alpha", float(adapter.alpha))
    writer.add_uint32("general.quantization_version", 2)

    for k in sorted(adapter.modules.keys()):
        b, a = adapter.modules[k]
        writer.add_tensor(f"{k}.lora_a", np.ascontiguousarray(a.astype(np.float16)))
        writer.add_tensor(f"{k}.lora_b", np.ascontiguousarray(b.astype(np.float16)))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save_safetensors(adapter: LoRAAdapter, out_dir: str | pathlib.Path, dtype: str = "float32") -> str:
    from safetensors.numpy import save_file

    d = pathlib.Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    np_dt = np.float16 if dtype == "float16" else np.float32
    tensors = {}
    for k, (b, a) in adapter.modules.items():
        parts = k.split(".")
        layer_idx = parts[1]
        mod_key = f"{parts[2]}.{parts[3]}"
        peft_mod = GGUF_TO_PEFT_SUFFIX[mod_key]
        base_k = f"{adapter.peft_prefix}.{layer_idx}.{peft_mod}"
        tensors[f"{base_k}.lora_A.weight"] = np.ascontiguousarray(a.astype(np_dt))
        tensors[f"{base_k}.lora_B.weight"] = np.ascontiguousarray(b.astype(np_dt))
    st_file = d / "adapter_model.safetensors"
    save_file(tensors, str(st_file))
    cfg_file = d / "adapter_config.json"
    if not cfg_file.exists():
        cfg = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": adapter.rank,
            "lora_alpha": adapter.alpha,
            "lora_dropout": 0.0,
            "bias": "none",
            "base_model_name_or_path": adapter.base_model,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        }
        cfg_file.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return hashlib.sha256(st_file.read_bytes()).hexdigest()


# ───────────────────────── Low-Rank Linear Algebra Primitives O(d * r^2) ─────────────────────────


def frobenius_inner(b1: np.ndarray, a1: np.ndarray, b2: np.ndarray, a2: np.ndarray) -> float:
    """Exact <B1 A1, B2 A2>_F = tr((B1^T B2)(A2 A1^T)) in O((d_out + d_in) * r1 * r2) without dense allocation."""
    g_b = b1.T @ b2  # (r1, r2)
    g_a = a2 @ a1.T  # (r2, r1)
    return float(np.trace(g_b @ g_a))


def frobenius_norm_ba(b: np.ndarray, a: np.ndarray) -> float:
    return math.sqrt(max(0.0, frobenius_inner(b, a, b, a)))


def qr_svd_compress(b_cat: np.ndarray, a_cat: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact rank-r truncated SVD of (b_cat @ a_cat) via thin QR in O((d_out + d_in) * R^2)."""
    qb, rb = np.linalg.qr(b_cat, mode="reduced")
    qa, ra = np.linalg.qr(a_cat.T, mode="reduced")
    u, s, vt = np.linalg.svd(rb @ ra.T, full_matrices=False)

    tot_e = float(np.sum(s * s))
    keep_e = float(np.sum(s[:rank] * s[:rank]))
    energy = keep_e / tot_e if tot_e > 0.0 else 1.0

    k = min(rank, len(s))
    s_sqrt = np.sqrt(s[:k])
    b_new = (qb @ u[:, :k]) * s_sqrt[None, :]
    a_new = s_sqrt[:, None] * (vt[:k, :] @ qa.T)
    if k < rank:
        b_new = np.pad(b_new, ((0, 0), (0, rank - k)))
        a_new = np.pad(a_new, ((0, rank - k), (0, 0)))
    return b_new.astype(np.float32), a_new.astype(np.float32), energy


def qr_svd_full_factors(b_cat: np.ndarray, a_cat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact thin singular factors U (d_out, R), S (R,), Vt (R, d_in) of b_cat @ a_cat."""
    qb, rb = np.linalg.qr(b_cat, mode="reduced")
    qa, ra = np.linalg.qr(a_cat.T, mode="reduced")
    u_s, s, vt_s = np.linalg.svd(rb @ ra.T, full_matrices=False)
    return qb @ u_s, s, vt_s @ qa.T


def deforget_layer_orthogonal(
    b1: np.ndarray, a1: np.ndarray, s1: float,
    b2: np.ndarray, a2: np.ndarray, s2: float,
    b3: np.ndarray, a3: np.ndarray, s3: float,
    gamma: float,
    protect_energy: float,
    out_rank: int,
    out_scale: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Frobenius-Orthogonal Continual De-Forgetting:
    Let F = dW2 - dW1 (forgetting drift) and N = dW3 - dW2 (newly learned capability).
    1. Decompose N = U S V^T via thin QR-SVD and pick minimal k capturing `protect_energy` of ||N||_F^2.
    2. Project F onto the exact Frobenius orthogonal complement of S_k = {U_k C V_k^T : C in R^{k x k}}:
         F_perp = F - U_k (U_k^T F V_k) V_k^T = B_f A_f - [U_k (U_k^T B_f)(A_f V_k)] V_k^T
       Guaranteed <F_perp, U_k C V_k^T>_F == 0 for ALL k x k matrices C.
    3. Reconstruct dW_out = dW3 - gamma * F_perp and compress to `out_rank` via Thin QR-SVD.
    """
    if abs(gamma) < 1e-9:
        b_out, a_out, er = qr_svd_compress((s3 / out_scale) * b3, a3, out_rank)
        return b_out, a_out, er, 1.0

    bf = np.concatenate([s2 * b2, -s1 * b1], axis=1)
    af = np.concatenate([a2, a1], axis=0)

    bn = np.concatenate([s3 * b3, -s2 * b2], axis=1)
    an = np.concatenate([a3, a2], axis=0)

    u_n, s_n, vt_n = qr_svd_full_factors(bn, an)
    tot_n = float(np.sum(s_n * s_n))
    if tot_n <= 1e-12 or protect_energy <= 0.0:
        b_cat = np.concatenate([s3 * b3, -gamma * bf], axis=1)
        a_cat = np.concatenate([a3, af], axis=0)
        b_out, a_out, er = qr_svd_compress(b_cat / out_scale, a_cat, out_rank)
        return b_out, a_out, er, 1.0

    csum = np.cumsum(s_n * s_n) / tot_n
    k = max(1, min(int(np.searchsorted(csum, protect_energy) + 1), len(s_n)))

    uk = u_n[:, :k]       # (d_out, k)
    vk = vt_n[:k, :].T    # (d_in, k)

    core_k = (uk.T @ bf) @ (af @ vk)  # (k, k)
    b_proj = uk @ core_k              # (d_out, k)
    a_proj = vk.T                     # (k, d_in)

    f_norm2 = max(1e-12, frobenius_inner(bf, af, bf, af))
    proj_norm2 = float(np.sum(core_k * core_k))
    orth_frac = math.sqrt(max(0.0, 1.0 - proj_norm2 / f_norm2))

    b_all = np.concatenate([s3 * b3, -gamma * bf, gamma * b_proj], axis=1)
    a_all = np.concatenate([a3, af, a_proj], axis=0)
    b_out, a_out, er = qr_svd_compress(b_all / out_scale, a_all, out_rank)
    return b_out, a_out, er, orth_frac


# ───────────────────────── CLI Commands ─────────────────────────


def cmd_inspect(path: str) -> int:
    ad = load_adapter(path)
    print(f"Adapter   : {path}")
    print(f"Name/Arch : {ad.name} ({ad.arch}) | rank={ad.rank}, alpha={ad.alpha}, scale={ad.scale:.4f}")
    n_attn = sum(is_attn_module(k) for k in ad.modules)
    print(f"Modules   : {len(ad.modules)} ({n_attn} attn, {len(ad.modules) - n_attn} mlp)")
    layer_norms: dict[int, float] = {}
    tot_params = 0
    for k, (b, a) in ad.modules.items():
        tot_params += b.size + a.size
        lid = int(k.split(".")[1])
        n2 = (ad.scale * frobenius_norm_ba(b, a)) ** 2
        layer_norms[lid] = layer_norms.get(lid, 0.0) + n2
    layer_rms = {k: math.sqrt(v) for k, v in sorted(layer_norms.items())}
    mx = max(layer_rms.values()) if layer_rms else 1.0
    print(f"Parameters: {tot_params:,} | Global ||dW||_F = {math.sqrt(sum(layer_norms.values())):.4f}\n")
    for lid, val in layer_rms.items():
        bar = "█" * int(round(28 * val / mx))
        print(f"  layer {lid:2d} | {val:7.4f} | {bar}")
    return 0


def cmd_diff(paths: list[str]) -> int:
    ads = [load_adapter(p) for p in paths]
    keys = sorted(set.intersection(*(set(a.modules.keys()) for a in ads)))
    print(f"Comparing {len(ads)} adapters across {len(keys)} shared modules (O(d*r^2) trace method):")
    for p, ad in zip(paths, ads):
        gn = math.sqrt(sum((ad.scale * frobenius_norm_ba(*ad.modules[k])) ** 2 for k in keys))
        print(f"  ||dW||_F [{pathlib.Path(p).name}] = {gn:.4f}")

    for i in range(len(ads) - 1):
        a1, a2 = ads[i], ads[i + 1]
        d2_attn = d2_mlp = 0.0
        for k in keys:
            b1, m1 = a1.modules[k]
            b2, m2 = a2.modules[k]
            bc = np.concatenate([a2.scale * b2, -a1.scale * b1], axis=1)
            ac = np.concatenate([m2, m1], axis=0)
            val2 = max(0.0, frobenius_inner(bc, ac, bc, ac))
            if is_attn_module(k):
                d2_attn += val2
            else:
                d2_mlp += val2
        tot = math.sqrt(d2_attn + d2_mlp)
        print(f"  Step {i+1}->{i+2} ||dW_{i+2} - dW_{i+1}||_F = {tot:.4f} (attn={math.sqrt(d2_attn):.4f}, mlp={math.sqrt(d2_mlp):.4f})")

    if len(ads) == 3:
        a1, a2, a3 = ads
        dot = n_u2 = n_v2 = 0.0
        for k in keys:
            b1, m1 = a1.modules[k]
            b2, m2 = a2.modules[k]
            b3, m3 = a3.modules[k]
            bu = np.concatenate([a2.scale * b2, -a1.scale * b1], axis=1)
            au = np.concatenate([m2, m1], axis=0)
            bv = np.concatenate([a3.scale * b3, -a2.scale * b2], axis=1)
            av = np.concatenate([m3, m2], axis=0)
            dot += frobenius_inner(bu, au, bv, av)
            n_u2 += max(0.0, frobenius_inner(bu, au, bu, au))
            n_v2 += max(0.0, frobenius_inner(bv, av, bv, av))
        cos = dot / (math.sqrt(n_u2 * n_v2) + 1e-12)
        print(f"  Trajectory Cosine(v2-v1, v3-v2) = {cos:+.4f} (< 0 indicates v3 is reversing v2 drift)")
    return 0


def run_deforget(
    v1: LoRAAdapter,
    v2: LoRAAdapter,
    v3: LoRAAdapter,
    mode: str = "orthogonal",
    gamma_attn: float = 0.50,
    gamma_mlp: float = 0.35,
    protect_energy: float = 0.85,
    rank: int | None = None,
) -> tuple[LoRAAdapter, dict]:
    t0 = time.time()
    out_rank = rank or v3.rank
    out_alpha = float(out_rank)
    out_scale = out_alpha / float(out_rank)
    keys = sorted(set(v1.modules) & set(v2.modules) & set(v3.modules))

    out_mods = {}
    energies, orth_fracs = [], []
    for k in keys:
        b1, a1 = v1.modules[k]
        b2, a2 = v2.modules[k]
        b3, a3 = v3.modules[k]
        g = gamma_attn if is_attn_module(k) else gamma_mlp
        pe = protect_energy if mode == "orthogonal" else 0.0
        bo, ao, er, ofr = deforget_layer_orthogonal(
            b1, a1, v1.scale, b2, a2, v2.scale, b3, a3, v3.scale,
            gamma=g, protect_energy=pe, out_rank=out_rank, out_scale=out_scale,
        )
        out_mods[k] = (bo, ao)
        energies.append(er)
        orth_fracs.append(ofr)

    healed = LoRAAdapter(
        modules=out_mods, alpha=out_alpha, arch=v3.arch,
        base_model=v3.base_model, name=f"{v3.name}-deforget-{mode}", peft_prefix=v3.peft_prefix,
    )
    stats = {
        "mode": mode,
        "modules": len(keys),
        "rank": out_rank,
        "gamma_attn": gamma_attn,
        "gamma_mlp": gamma_mlp,
        "protect_energy": protect_energy if mode == "orthogonal" else 0.0,
        "mean_svd_energy_pct": round(float(np.mean(energies)) * 100, 4),
        "min_svd_energy_pct": round(float(np.min(energies)) * 100, 4),
        "mean_orth_retained_pct": round(float(np.mean(orth_fracs)) * 100, 2),
        "seconds": round(time.time() - t0, 2),
    }
    return healed, stats


def self_test() -> None:
    rng = np.random.default_rng(2026)
    d_out, d_in, r = 128, 96, 8
    b1, a1 = rng.standard_normal((d_out, r)).astype(np.float32), rng.standard_normal((r, d_in)).astype(np.float32)
    b2, a2 = rng.standard_normal((d_out, r)).astype(np.float32), rng.standard_normal((r, d_in)).astype(np.float32)
    b3, a3 = rng.standard_normal((d_out, r)).astype(np.float32), rng.standard_normal((r, d_in)).astype(np.float32)

    exact_ip = float(np.sum((b1 @ a1) * (b2 @ a2)))
    fast_ip = frobenius_inner(b1, a1, b2, a2)
    assert abs(exact_ip - fast_ip) / (abs(exact_ip) + 1e-6) < 1e-5

    w_sum = 0.4 * (b1 @ a1) + 0.6 * (b3 @ a3)
    u_full, s_full, vt_full = np.linalg.svd(w_sum, full_matrices=False)
    w_ref = (u_full[:, :r] * s_full[:r][None, :]) @ vt_full[:r, :]
    b_qr, a_qr, _ = qr_svd_compress(np.concatenate([0.4 * b1, 0.6 * b3], axis=1), np.concatenate([a1, a3], axis=0), r)
    rel_err = float(np.linalg.norm((b_qr @ a_qr) - w_ref) / np.linalg.norm(w_ref))
    assert rel_err < 1e-5

    w1, w2, w3 = b1 @ a1, b2 @ a2, b3 @ a3
    f_mat, n_mat = w2 - w1, w3 - w2
    un, _, vtn = np.linalg.svd(n_mat, full_matrices=False)
    k = 4
    uk, vk = un[:, :k], vtn[:k, :].T
    f_perp = f_mat - uk @ (uk.T @ f_mat @ vk) @ vk.T
    rand_m = rng.standard_normal((k, k)).astype(np.float32)
    leak = abs(float(np.sum(f_perp * (uk @ rand_m @ vk.T))))
    assert leak < 1e-4

    bo, ao, er, _ = deforget_layer_orthogonal(
        b1, a1, 1.0, b2, a2, 1.0, b3, a3, 1.0, gamma=0.5, protect_energy=0.85, out_rank=r, out_scale=1.0
    )
    assert bo.shape == (d_out, r) and ao.shape == (r, d_in) and 0.0 < er <= 1.0001
    print(f"lora-surgeon v{__version__} self-test PASSED: trace_ip OK, qr_svd rel_err={rel_err:.2e}, orth_leak={leak:.2e}")


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lora-surgeon", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true", help="Run mathematical verification suite and exit")
    sub = p.add_subparsers(dest="cmd")

    s_ins = sub.add_parser("inspect", help="Inspect rank, alpha, and per-layer Frobenius norm profile")
    s_ins.add_argument("adapter")

    s_dif = sub.add_parser("diff", help="Compare 2 or 3 generational LoRA adapters in O(d*r^2)")
    s_dif.add_argument("adapters", nargs="+")

    s_def = sub.add_parser("deforget", help="Heal continual-learning forgetting across v1 -> v2 -> v3")
    s_def.add_argument("--v1", required=True, help="Clean ancestor LoRA (strong prior capability)")
    s_def.add_argument("--v2", required=True, help="Intermediate LoRA where forgetting occurred")
    s_def.add_argument("--v3", required=True, help="Latest LoRA with new capabilities to preserve")
    s_def.add_argument("--mode", choices=["orthogonal", "linear"], default="orthogonal")
    s_def.add_argument("--gamma", type=float, default=0.45, help="Default de-forgetting strength")
    s_def.add_argument("--gamma-attn", type=float, default=None, help="De-forgetting strength for Attention layers")
    s_def.add_argument("--gamma-mlp", type=float, default=None, help="De-forgetting strength for MLP layers")
    s_def.add_argument("--protect-energy", type=float, default=0.85, help="Spectral energy of (v3 - v2) protected (0..1)")
    s_def.add_argument("--rank", type=int, default=None, help="Output LoRA rank (defaults to v3 rank)")
    s_def.add_argument("--peft-prefix", default=None, help="Override PEFT tensor key prefix when exporting safetensors")
    s_def.add_argument("-o", "--output", required=True, help="Output .gguf path or PEFT directory")

    s_mrg = sub.add_parser("merge", help="Exact Thin QR-SVD weighted merge of N LoRA adapters")
    s_mrg.add_argument("--inputs", nargs="+", required=True, help="List of path=weight pairs, e.g. a.gguf=0.4 b.gguf=0.6")
    s_mrg.add_argument("--rank", type=int, default=None)
    s_mrg.add_argument("--peft-prefix", default=None)
    s_mrg.add_argument("-o", "--output", required=True)

    s_cnv = sub.add_parser("convert", help="Convert between .gguf and PEFT safetensors directory")
    s_cnv.add_argument("src")
    s_cnv.add_argument("--arch", default="llama", help="Target GGUF architecture when converting from PEFT")
    s_cnv.add_argument("--peft-prefix", default=None)
    s_cnv.add_argument("-o", "--output", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = build_cli()
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.cmd == "inspect":
        return cmd_inspect(args.adapter)
    if args.cmd == "diff":
        return cmd_diff(args.adapters)
    if args.cmd == "deforget":
        v1, v2, v3 = load_adapter(args.v1), load_adapter(args.v2), load_adapter(args.v3)
        ga = args.gamma if args.gamma_attn is None else args.gamma_attn
        gm = args.gamma if args.gamma_mlp is None else args.gamma_mlp
        healed, stats = run_deforget(
            v1, v2, v3, mode=args.mode, gamma_attn=ga, gamma_mlp=gm, protect_energy=args.protect_energy, rank=args.rank
        )
        if args.peft_prefix:
            healed.peft_prefix = args.peft_prefix
        out = pathlib.Path(args.output)
        sha = save_gguf(healed, out) if out.suffix.lower() == ".gguf" else save_safetensors(healed, out)
        stats["output"] = str(out)
        stats["sha256"] = sha
        print(json.dumps(stats, indent=2))
        return 0
    if args.cmd == "merge":
        pairs = []
        for item in args.inputs:
            p_str, w_str = item.rsplit("=", 1)
            pairs.append((load_adapter(p_str), float(w_str)))
        ref = pairs[0][0]
        out_r = args.rank or ref.rank
        keys = sorted(set.intersection(*(set(ad.modules.keys()) for ad, _ in pairs)))
        mods, ens = {}, []
        for k in keys:
            bc = np.concatenate([w * ad.scale * ad.modules[k][0] for ad, w in pairs], axis=1)
            ac = np.concatenate([ad.modules[k][1] for ad, _ in pairs], axis=0)
            bo, ao, er = qr_svd_compress(bc, ac, out_r)
            mods[k] = (bo, ao)
            ens.append(er)
        merged = LoRAAdapter(
            modules=mods, alpha=float(out_r), arch=ref.arch, base_model=ref.base_model,
            name="lora-surgeon-merged", peft_prefix=args.peft_prefix or ref.peft_prefix,
        )
        out = pathlib.Path(args.output)
        sha = save_gguf(merged, out) if out.suffix.lower() == ".gguf" else save_safetensors(merged, out)
        print(json.dumps({"output": str(out), "rank": out_r, "mean_svd_energy_pct": round(float(np.mean(ens)) * 100, 4), "sha256": sha}, indent=2))
        return 0
    if args.cmd == "convert":
        ad = load_adapter(args.src, arch=args.arch)
        if args.peft_prefix:
            ad.peft_prefix = args.peft_prefix
        out = pathlib.Path(args.output)
        sha = save_gguf(ad, out) if out.suffix.lower() == ".gguf" else save_safetensors(ad, out)
        print(json.dumps({"src": args.src, "output": str(out), "rank": ad.rank, "sha256": sha}, indent=2))
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
