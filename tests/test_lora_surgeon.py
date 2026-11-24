from __future__ import annotations

import pathlib
import numpy as np
import lora_surgeon as ls


def _synthetic_adapter(seed: int, rank: int = 8) -> ls.LoRAAdapter:
    rng = np.random.default_rng(seed)
    mods = {
        "blk.0.attn_q.weight": (
            rng.standard_normal((64, rank)).astype(np.float32),
            rng.standard_normal((rank, 48)).astype(np.float32),
        ),
        "blk.0.ffn_down.weight": (
            rng.standard_normal((48, rank)).astype(np.float32),
            rng.standard_normal((rank, 96)).astype(np.float32),
        ),
    }
    return ls.LoRAAdapter(modules=mods, alpha=float(rank), arch="llama", base_model="test-base", name=f"gen{seed}")


def test_trace_inner_product_exactness():
    rng = np.random.default_rng(7)
    b1, a1 = rng.standard_normal((96, 8)).astype(np.float32), rng.standard_normal((8, 64)).astype(np.float32)
    b2, a2 = rng.standard_normal((96, 12)).astype(np.float32), rng.standard_normal((12, 64)).astype(np.float32)
    dense = float(np.sum((b1 @ a1) * (b2 @ a2)))
    fast = ls.frobenius_inner(b1, a1, b2, a2)
    assert np.isclose(dense, fast, rtol=1e-5)


def test_qr_svd_matches_dense_svd():
    rng = np.random.default_rng(11)
    b = rng.standard_normal((120, 16)).astype(np.float32)
    a = rng.standard_normal((16, 80)).astype(np.float32)
    u, s, vt = np.linalg.svd(b @ a, full_matrices=False)
    dense_r8 = (u[:, :8] * s[:8][None, :]) @ vt[:8, :]
    b_qr, a_qr, energy = ls.qr_svd_compress(b, a, rank=8)
    assert 0.0 < energy <= 1.0
    assert np.linalg.norm((b_qr @ a_qr) - dense_r8) / np.linalg.norm(dense_r8) < 1e-5


def test_orthogonal_projection_zero_leak():
    rng = np.random.default_rng(42)
    d_out, d_in, r = 64, 48, 4
    w1 = rng.standard_normal((d_out, r)).astype(np.float32) @ rng.standard_normal((r, d_in)).astype(np.float32)
    w2 = w1 + rng.standard_normal((d_out, r)).astype(np.float32) @ rng.standard_normal((r, d_in)).astype(np.float32)
    w3 = w2 + rng.standard_normal((d_out, r)).astype(np.float32) @ rng.standard_normal((r, d_in)).astype(np.float32)
    u, _, vt = np.linalg.svd(w3 - w2, full_matrices=False)
    k = 3
    uk, vk = u[:, :k], vt[:k, :].T
    f = w2 - w1
    f_perp = f - uk @ (uk.T @ f @ vk) @ vk.T
    c = rng.standard_normal((k, k)).astype(np.float32)
    assert abs(float(np.sum(f_perp * (uk @ c @ vk.T)))) < 1e-4


def test_gguf_and_safetensors_roundtrip(tmp_path: pathlib.Path):
    ad = _synthetic_adapter(1, rank=8)
    gguf_p = tmp_path / "v1.gguf"
    peft_d = tmp_path / "v1_peft"
    ls.save_gguf(ad, gguf_p)
    reloaded_gguf = ls.load_adapter(gguf_p)
    assert set(reloaded_gguf.modules.keys()) == set(ad.modules.keys())
    for k in ad.modules:
        b0, a0 = ad.modules[k]
        b1, a1 = reloaded_gguf.modules[k]
        assert np.allclose(b0 @ a0, b1 @ a1, rtol=5e-3, atol=5e-3)

    ls.save_safetensors(reloaded_gguf, peft_d)
    reloaded_peft = ls.load_adapter(peft_d)
    for k in ad.modules:
        b1, a1 = reloaded_gguf.modules[k]
        b2, a2 = reloaded_peft.modules[k]
        assert np.allclose(b1 @ a1, b2 @ a2, rtol=1e-5, atol=1e-5)


def test_cli_end_to_end_deforget_and_merge(tmp_path: pathlib.Path):
    v1, v2, v3 = _synthetic_adapter(1), _synthetic_adapter(2), _synthetic_adapter(3)
    p1, p2, p3 = tmp_path / "v1.gguf", tmp_path / "v2.gguf", tmp_path / "v3.gguf"
    out_def = tmp_path / "healed.gguf"
    out_mrg = tmp_path / "merged.gguf"
    ls.save_gguf(v1, p1)
    ls.save_gguf(v2, p2)
    ls.save_gguf(v3, p3)

    assert ls.main(["inspect", str(p3)]) == 0
    assert ls.main(["diff", str(p1), str(p2), str(p3)]) == 0
    assert ls.main([
        "deforget", "--v1", str(p1), "--v2", str(p2), "--v3", str(p3),
        "--mode", "orthogonal", "--gamma-attn", "0.5", "--gamma-mlp", "0.25",
        "-o", str(out_def),
    ]) == 0
    assert out_def.exists() and out_def.stat().st_size > 0
    assert ls.main(["merge", "--inputs", f"{p1}=0.4", f"{p3}=0.6", "-o", str(out_mrg)]) == 0
    assert out_mrg.exists() and out_mrg.stat().st_size > 0
