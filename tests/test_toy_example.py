from __future__ import annotations

import importlib.util
import pathlib

EXAMPLE = pathlib.Path(__file__).resolve().parent.parent / "examples" / "toy_continual_forgetting.py"


def test_toy_example_runs_and_writes_adapters(tmp_path: pathlib.Path, capsys):
    spec = importlib.util.spec_from_file_location("toy_example", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.main(["--seeds", "0", "--gammas", "0.4", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "| Gen-3 |" in out and "| orthogonal gamma=0.4 |" in out
    for g in (1, 2, 3):
        assert (tmp_path / f"gen{g}.gguf").stat().st_size > 0
