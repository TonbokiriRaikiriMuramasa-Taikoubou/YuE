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

sys.path.insert(0, str(Path(__file__).resolve().parent))          # the runner
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # yue2.protocol (no torch)

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
    assert big["memory_budget_gib"] == 23.6 and big["offload_ar"] is False
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


def test_gpu_memory_helpers_degrade_without_cuda():
    memory = yk.gpu_memory()
    assert isinstance(memory, dict) and "cuda" in memory
    released = yk.release_gpu_memory()
    assert isinstance(released, dict) and "cuda" in released


def _frames(exception):
    import traceback

    names = []
    for entry in traceback.extract_tb(exception.__traceback__) if exception.__traceback__ else []:
        names.append(entry.name)
    return names


class _FakePipe:
    """Stands in for YuE2Pipeline: the handle only flips .backend and closes it."""

    def __init__(self):
        self.backend = "torch"
        self.closed = 0

    def close(self):
        self.closed += 1


def _t4_report():
    return {"cuda_available": True, "current_device": "cuda:0", "torch": "2.10.0",
            "bf16_supported": True, "bf16_native": False,
            "devices": [{"name": "Tesla T4", "memory_gib": 15.9, "free_gib": 15.9,
                         "compute_capability": [7, 5]}]}


def _ampere_report():
    return {"cuda_available": True, "current_device": "cuda:0", "torch": "2.10.0",
            "bf16_supported": True, "bf16_native": True,
            "devices": [{"name": "NVIDIA A10G", "memory_gib": 23.6, "free_gib": 23.6,
                         "compute_capability": [8, 6]}]}


def test_backend_is_chosen_from_compute_capability(monkeypatch):
    built = []

    def fake_build(report, options, backend, generation_config=None):
        built.append(backend)
        pipe = _FakePipe()
        pipe.backend = backend
        return pipe

    monkeypatch.setattr(yk, "build_pipeline", fake_build)
    options = yk.Options(seconds=60)

    t4 = yk.PipelineHandle(_t4_report(), options)          # cc 7.5: flash is impossible
    assert t4.backend == "torch-eager" and t4.level == 1
    assert "no FlashAttention" in t4.backend_reason

    ampere = yk.PipelineHandle(_ampere_report(), options)  # cc 8.6: fast path stays
    assert ampere.backend == "torch" and ampere.level == 0

    explicit = yk.PipelineHandle(_t4_report(), yk.Options(seconds=60, backend="torch"))
    assert explicit.backend == "torch" and "explicit" in explicit.backend_reason

    assert built == ["torch-eager", "torch", "torch"]


def test_downgrade_is_in_place_and_releases_the_failure(monkeypatch):
    monkeypatch.setattr(yk, "build_pipeline",
                        lambda report, options, backend, generation_config=None: _FakePipe())
    handle = yk.PipelineHandle(_ampere_report(), yk.Options(seconds=60))
    pipe = handle.pipe
    error = RuntimeError("FlashAttention only supports Ampere GPUs or newer.")
    try:
        raise error
    except RuntimeError as exc:
        assert handle.downgrade(exc) is True
        assert exc.__traceback__ is None, "the traceback pins the failed model in VRAM"

    assert handle.pipe is pipe, "downgrade must reuse the verified pipeline, not rebuild it"
    assert handle.pipe.backend == "torch-eager" and handle.backend == "torch-eager"
    assert pipe.closed == 1, "the GPU copy must be dropped so the stage reloads it"


def test_second_downgrade_step_restricts_attention(monkeypatch):
    monkeypatch.setattr(yk, "build_pipeline",
                        lambda report, options, backend, generation_config=None: _FakePipe())
    calls = []
    monkeypatch.setattr(yk.PipelineHandle, "_disable_fused_sdpa",
                        staticmethod(lambda: calls.append("disabled")))
    handle = yk.PipelineHandle(_ampere_report(), yk.Options(seconds=60))
    handle.downgrade(RuntimeError("FlashAttention only supports Ampere GPUs or newer."))
    assert handle.level == 1 and calls == []
    handle.downgrade(RuntimeError("No available kernel for bf16 scaled_dot_product_attention"))
    assert handle.level == 2 and calls == ["disabled"]
    assert handle.downgrade(RuntimeError("still broken")) is False


def test_downgrade_leaves_explicit_backends_alone(monkeypatch):
    monkeypatch.setattr(yk, "build_pipeline",
                        lambda report, options, backend, generation_config=None: _FakePipe())
    handle = yk.PipelineHandle(_ampere_report(), yk.Options(seconds=60, backend="torch"))
    error = RuntimeError("FlashAttention only supports Ampere GPUs or newer.")
    try:
        raise error
    except RuntimeError as exc:
        assert handle.downgrade(exc) is False


def test_failed_stage_drops_its_traceback():
    """Jupyter keeps the last traceback; its frames would pin GPU tensors."""

    class Handle:
        pipe = "pipe"

        def downgrade(self, exc):
            return False

        def release_memory(self):
            self.released = True

    handle = Handle()

    def stage(pipe):
        raise RuntimeError("CUDA out of memory. Tried to allocate 722.00 MiB")

    try:
        yk.run_with_retry(handle, stage)
    except RuntimeError as exc:
        # the deep frames (module/param locals that pin GPU tensors) must be gone
        names = _frames(exc)
        assert "stage" not in names, names
        assert len(names) <= 3, names
        assert "out of memory" in str(exc)
    else:
        raise AssertionError("the OOM must propagate")
    assert handle.released is True

    # a non-OOM failure must not trigger a memory sweep, but still loses its frames
    handle2 = Handle()

    def stage2(pipe):
        raise ValueError("bad request")

    try:
        yk.run_with_retry(handle2, stage2)
    except ValueError as exc:
        assert "stage2" not in _frames(exc), _frames(exc)
    assert not hasattr(handle2, "released")


def test_import_probe_shape():
    probe = yk._import_probe()
    for name in ("numpy", "scipy", "sklearn", "transformers", "torch",
                 "transformers.GenerationMixin"):
        assert name in probe, name
    assert all(isinstance(value, str) for value in probe.values())


def test_verdict_rejects_an_inconsistent_python_environment():
    base = {"cuda_available": True, "current_device": "cuda:0", "torch": "2.10.0",
            "bf16_supported": True, "bf16_native": True, "ram_gib": 16,
            "devices": [{"name": "Tesla T4", "memory_gib": 15.9, "free_gib": 15.9,
                         "compute_capability": [7, 5]}],
            "disk": {"working (/kaggle/working)": {"free_gib": 40, "path": "x"}},
            "internet": {"huggingface.co": True}}
    broken = yk.verdict({**base, "runtime_imports": {
        "numpy": "2.2.6",
        "scipy": "FAILED: ImportError: cannot import name '_center' from 'numpy._core.umath'",
        "transformers.GenerationMixin": "FAILED: ImportError: ..."}})
    assert broken["ok"] is False
    assert any("inconsistent" in line for line in broken["checks"])
    assert any("INSTALL_MODE='image'" in line for line in broken["checks"])
    healthy = yk.verdict({**base, "runtime_imports": {"numpy": "2.3.1", "scipy": "1.17.0",
                                                      "transformers.GenerationMixin": "OK"}})
    assert healthy["ok"] is True
    # an empty probe (older callers) must not block anything
    assert yk.verdict(dict(base))["ok"] is True


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


def _run_settings_cell(source, **overrides):
    """Run cell 1 (idea) and then the given settings cell, as the notebook does."""
    import contextlib
    import io
    import pathlib
    import re

    for key, value in overrides.items():
        source, count = re.subn(rf"^{key} = .*$", f"{key} = {value!r}", source,
                                count=1, flags=re.M)
        assert count == 1, f"{key} not found in the settings cell"
    idea = next(src for src in _notebook_cells() if "use_preset" in src and "STYLE = (" in src)
    namespace = {"__name__": "config_cell", "Path": pathlib.Path}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(idea, "idea_cell", "exec"), namespace)
        exec(compile(source, "settings_cell", "exec"), namespace)
    return namespace, output.getvalue()


def _notebook_cells(kind="code"):
    import json
    notebook = json.loads((Path(__file__).parent / "YuE2_Kaggle.ipynb").read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == kind]


def test_notebook_cells_are_valid_python():
    import compileall  # noqa: F401 - documentation only
    checked = 0
    for index, source in enumerate(_notebook_cells()):
        if any(line.startswith("!") or line.startswith("%") for line in source.splitlines()):
            continue                                    # IPython magics: not plain Python
        compile(source, f"notebook-cell-{index}", "exec")
        checked += 1
    assert checked >= 6


def _run_config_cells(**overrides):
    settings = next(src for src in _notebook_cells() if "OUTPUT_ID" in src and "QUICK_PREVIEW" in src)
    return _run_settings_cell(settings, **overrides)


def test_notebook_idea_cell_explains_style_and_lyrics():
    """Cell 1 must run before anything is installed and explain STYLE/LYRICS."""
    import contextlib
    import io

    namespace, text = _run_config_cells()
    assert "✅ [1/9 アイディア] OK" in text, text
    assert "🎼" in text and "🎤" in text
    assert "ネオンが路地に消えていく" in text            # default lyrics translation
    assert "✅ [2/9 設定] OK" in text, text
    assert "OUTPUT_ID を自動採番" in text                 # stable, automatic output folder
    with contextlib.redirect_stdout(io.StringIO()):
        assert namespace["mark"]("probe", False, "x") is False

    describe = namespace["describe_style"]
    translated = describe("English, warm piano pop, expressive female voice, 88 BPM")
    assert "英語" in translated and "温かいピアノポップス" in translated
    assert "表現力豊かな女性ボーカル" in translated and "88 BPM" in translated
    # Japanese input passes through, and unknown English words are kept
    assert describe("Japanese, city pop, ドラム") == "日本語 / シティポップス / ドラム"
    assert "conductor" in describe("orchestral, conductor")


def test_notebook_presets_and_quick_preview():
    presets = ["citypop_ja", "ballad_ja", "lofi_en", "rock_en"]
    namespace, _ = _run_config_cells()
    assert set(namespace["PRESETS"]) == set(presets)
    import contextlib
    import io

    for name in presets:
        with contextlib.redirect_stdout(io.StringIO()):
            namespace["use_preset"](name)
        style, lyrics = namespace["PRESETS"][name]["style"], namespace["PRESETS"][name]["lyrics"]
        assert namespace["STYLE"] == style and namespace["LYRICS"] == lyrics
        assert "[Verse]" in lyrics and "[Chorus]" in lyrics
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            namespace["use_preset"]("nope")
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown preset must raise")

    # the same settings must map to the same output folder, different ones must not
    def output_id(**overrides):
        return _run_config_cells(**overrides)[0]["OUTPUT_ID"]

    assert output_id() == output_id()
    assert output_id(ODE_STEPS=8) != output_id(ODE_STEPS=32)

    quick = next(src for src in _notebook_cells() if "QUICK_PREVIEW" in src)
    quick_source = quick.replace("QUICK_PREVIEW = False", "QUICK_PREVIEW = True")
    fast, text = _run_settings_cell(quick_source)
    assert (fast["SECONDS"], fast["ODE_STEPS"], fast["PLAN_MAX_TOKENS"]) == (60.0, 8, 512)
    assert "QUICK_PREVIEW=True" in text


def test_notebook_embeds_the_driver_verbatim():
    source = next(src for src in _notebook_cells() if src.startswith("%%writefile"))
    embedded = source.split("\n", 1)[1]
    driver = (Path(__file__).parent / "yue2_kaggle.py").read_text(encoding="utf-8")
    assert embedded.rstrip("\n") == driver.rstrip("\n")
    assert yk.DRIVER_BUILD in embedded


def test_run_with_retry_retries_a_graph_refusal_but_not_a_real_error():
    seen = []

    class Handle:
        pipe = "rebuilt"

        def downgrade(self, exc):
            seen.append(type(exc).__name__)
            return True

    for error in (ValueError("FlashAttention only supports Ampere GPUs"),
                  RuntimeError("cuDNN attention is unavailable")):
        calls = {"n": 0}

        def stage(pipe, error=error, calls=calls):
            calls["n"] += 1
            if calls["n"] == 1:
                raise error
            return "recovered"

        assert yk.run_with_retry(Handle(), stage) == "recovered"
    assert seen == ["ValueError", "RuntimeError"]

    class Stubborn(Handle):
        def downgrade(self, exc):
            return False

    for error in (RuntimeError("CUDA out of memory"), ValueError("bad request")):
        def stage(pipe, error=error):
            raise error

        try:
            yk.run_with_retry(Stubborn(), stage)
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError(f"{error!r} must not be swallowed")


class _MonkeyPatch:
    """The small part of pytest's monkeypatch fixture this file needs, so the
    checks also run with a plain ``python kaggle/test_yue2_kaggle_logic.py``."""

    def __init__(self):
        self._undo = []

    def setattr(self, target, name, value, raising=True):
        had = hasattr(target, name)
        old = getattr(target, name, None)
        setattr(target, name, value)
        self._undo.append((target, name, had, old))

    def undo(self):
        for target, name, had, old in reversed(self._undo):
            if had:
                setattr(target, name, old)
            else:
                delattr(target, name)
        self._undo.clear()


if __name__ == "__main__":
    import inspect

    failures = 0
    for name, function in sorted(globals().items()):
        if not (name.startswith("test_") and callable(function)):
            continue
        patch = _MonkeyPatch()
        kwargs = {"monkeypatch": patch} if "monkeypatch" in inspect.signature(function).parameters else {}
        try:
            function(**kwargs)
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
        except Exception as exc:                       # noqa: BLE001 - report and continue
            failures += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"pass {name}")
        finally:
            patch.undo()
    raise SystemExit(1 if failures else 0)
