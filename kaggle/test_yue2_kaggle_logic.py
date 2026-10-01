#!/usr/bin/env python3
"""Dependency-free checks for the Kaggle runner's pure logic.

These never import torch or the model, so they run anywhere (Kaggle CPU cells,
CI, a laptop): ``python kaggle/test_yue2_kaggle_logic.py`` or ``pytest kaggle``.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yue2_kaggle as yk  # noqa: E402


def test_frame_geometry():
    assert yk.FRAME_SECONDS == 1920 / 48000          # 40 ms -> 25 fps, from YuE2-Vae
    assert yk.frames_for_seconds(120) == 3000
    assert yk.frames_for_seconds(0.001) == 1         # never zero frames
    assert abs(yk.seconds_for_frames(3000) - 120.0) < 1e-9
    # a full-length request stays inside the model context
    assert yk.frames_for_seconds(360) == 9000


def test_hms():
    assert yk.hms(45) == "45s"
    assert yk.hms(300) == "5.0 min"
    assert yk.hms(7200) == "2.00 h"
    assert yk.hms(None) == "n/a"


def test_options_validation_and_budgets():
    options = yk.Options(seconds=120, cot="melody")
    assert options.frames == 3000
    assert options.semantic_sampling()["max_tokens"] == 3000 + options.extra_semantic_margin
    short = yk.Options(seconds=1, cot="off")         # 25 frames < the release min_tokens=200
    sampling = short.semantic_sampling()
    assert sampling["max_tokens"] < 200 and 0 < sampling["min_tokens"] <= sampling["max_tokens"]
    for bad in ("melodies", ""):
        try:
            yk.Options(cot=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"cot={bad!r} should be rejected")
    try:
        yk.Options(cot="off", abc="X:1\nK:C\nCDEF|")
    except ValueError:
        pass
    else:
        raise AssertionError("abc with cot=off should be rejected")


def test_pipeline_settings_follow_vram():
    big = yk.pipeline_settings({"device": {"memory_gib": 23.6}})
    assert big["memory_budget_gib"] == 23.1 and big["offload_ar"] is False
    assert big["vae_core_frames"] == 1024 and big["quantization"] == "none"
    small = yk.pipeline_settings({"device": {"memory_gib": 11.0}})
    assert small["offload_ar"] is True and small["vae_core_frames"] == 512
    # the pipeline reserves 2 GiB and refuses budgets below that
    assert small["memory_budget_gib"] > 2.0


def _rates():
    return {"plan_tps": 8.0, "semantic_tps": 12.0, "nf_per_frame_seconds": 0.02,
            "nf_fixed_per_chunk_seconds": 30.0, "vae_seconds_per_frame": 0.004}


def test_predict_scales_with_length_and_steps():
    measured = _rates()
    short = yk.predict(measured, yk.Options(seconds=60, ode_steps=32))
    long = yk.predict(measured, yk.Options(seconds=120, ode_steps=32))
    cheap = yk.predict(measured, yk.Options(seconds=60, ode_steps=8))
    assert cheap["flow_matching"] < short["flow_matching"] < long["flow_matching"]
    ratio = short["flow_matching"] / cheap["flow_matching"]
    assert 3.5 < ratio < 4.5                         # 32 steps vs 8 steps
    for stages in (short, long, cheap):
        total = sum(stages[key] for key in ("plan", "semantic", "flow_matching", "decode"))
        assert abs(stages["total"] - total) < 1e-6
    # chunking: a six-minute song is one chunk as long as the prefix is short
    minutes = yk.predict(measured, yk.Options(seconds=360))
    assert minutes["frames"] == 9000 and minutes["chunks"] == 1


def test_predict_skips_planning_when_a_score_is_given():
    measured = _rates()
    supplied = yk.predict(measured, yk.Options(seconds=60, cot="melody", abc="X:1\nK:C\nCDEF|"))
    assert supplied["plan"] == 0.0
    assert yk.predict(measured, yk.Options(seconds=60, cot="off"))["plan"] == 0.0
    planned = yk.predict(measured, yk.Options(seconds=60, cot="full"))
    assert planned["plan"] > 0 and planned["plan_tokens_assumed"] <= 4096


def test_budget_gate():
    measured = _rates()
    options = yk.Options(seconds=180, ode_steps=32, budget_minutes=30, safety_minutes=5)
    stages = yk.predict(measured, options)
    assert stages["deadline_seconds"] == 25 * 60
    assert stages["fits_budget"] is (stages["total"] <= 25 * 60)


def test_find_local_models_reads_configs():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mot = root / "dataset" / "YuE2-3B"
        vae = root / "dataset" / "YuE2-Vae"
        mot.mkdir(parents=True)
        vae.mkdir(parents=True)
        (mot / "config.json").write_text(json.dumps({"model_type": "yue2"}), encoding="utf-8")
        (mot / "qwen.tiktoken").write_bytes(b"x")
        (vae / "config.json").write_text(json.dumps({"model_type": "yue2_vae"}), encoding="utf-8")
        (vae / "model.safetensors").write_bytes(b"x")
        found = yk.find_local_models((root,))
        assert found == (str(mot), str(vae))


def test_verdict_shapes():
    no_gpu = yk.verdict({"cuda_available": False})
    assert no_gpu["ok"] is False and "T4" in no_gpu["checks"][0]
    p100 = yk.verdict({"cuda_available": True, "current_device": "cuda:0", "torch": "2.10.0",
                       "bf16_supported": False,
                       "devices": [{"name": "Tesla P100-PCIE-16GB", "memory_gib": 15.9,
                                    "free_gib": 15.9, "compute_capability": [6, 0]}],
                       "disk": {}})
    assert p100["ok"] is False and any("P100" in line for line in p100["checks"])
    t4 = yk.verdict({"cuda_available": True, "current_device": "cuda:0", "torch": "2.10.0",
                     "bf16_supported": True, "bf16_native": False, "ram_gib": 16,
                     "devices": [{"name": "Tesla T4", "memory_gib": 15.9, "free_gib": 15.9,
                                  "compute_capability": [7, 5]}],
                     "disk": {"working (/kaggle/working)": {"free_gib": 40, "path": "x"}}})
    assert t4["ok"] is True and any("emulated" in line for line in t4["warnings"])


def test_verdict_needs_network_or_a_local_model():
    base = {"cuda_available": True, "current_device": "cuda:0", "torch": "2.10.0",
            "bf16_supported": True, "bf16_native": True, "ram_gib": 16,
            "devices": [{"name": "Tesla T4", "memory_gib": 15.9, "free_gib": 15.9,
                         "compute_capability": [7, 5]}],
            "disk": {"working (/kaggle/working)": {"free_gib": 40, "path": "x"}}}
    offline = yk.verdict({**base, "internet": {"huggingface.co": False}})
    assert offline["ok"] is False and any("Internet" in line for line in offline["checks"])
    with_local = yk.verdict({**base, "internet": {"huggingface.co": False},
                             "local_model_dir": "/kaggle/input/yue2/YuE2-3B"})
    assert with_local["ok"] is True
    assert any("unreachable" in line for line in with_local["warnings"])
    online = yk.verdict({**base, "internet": {"huggingface.co": True}})
    assert online["ok"] is True and not online["warnings"]


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
            else:
                print(f"pass {name}")
    raise SystemExit(1 if failures else 0)
