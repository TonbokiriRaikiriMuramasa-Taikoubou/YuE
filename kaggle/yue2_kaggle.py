#!/usr/bin/env python3
"""YuE2 on a free-tier Kaggle notebook: preflight, measured ETA, staged generation.

This module is a thin runner around the *unmodified* upstream ``yue2`` package.
It adds the parts that a 16 GB, non-native-BF16, 12-hour-session environment
needs:

* **Preflight.** Reports device, compute capability, VRAM, RAM, disk, and the
  pipeline's own BF16 gate, then derives safe ``YuE2Pipeline`` settings for this
  device instead of assuming the published 24 GB / native-BF16 baseline.
* **Honest estimates.** A one-to-two minute micro-benchmark of the real model
  turns the "24 GB BF16 GPU" guidance into a concrete per-song ETA for *this*
  GPU, so a run is never started blind.
* **Staged generation with resume.** plan -> semantic -> flow matching ->
  decode. Every stage is cached on disk, so a Kaggle session limit, a 20-minute
  interactive idle timeout, or a crashed kernel never throws away completed
  work: re-run the notebook and it continues.
* **Safety margins.** A wall-clock deadline is passed to the pipeline's
  ``cancelled`` hook so a stage stops cleanly before the session dies.

Nothing here changes model math, sampling defaults, or the artifact protocol.
The stages call ``YuE2Pipeline.plan / generate_semantic / synthesize / decode``
with the release configuration. Two clearly marked helpers reach for the
module-level functions the pipeline itself calls (``yue2.nar``) so flow matching
can checkpoint one chunk at a time; each has a public-API fallback.

Usage inside a Kaggle notebook (after installing the repository)::

    import yue2_kaggle as yk
    env = yk.environment(); yk.print_report(env)
    opts = yk.Options(outdir=Path("/kaggle/working/outputs/demo"), seconds=120)
    with yk.open_pipeline(env, opts) as handle:
        rates = yk.measure(handle)
        yk.print_estimate(yk.predict(rates, opts), opts)
        yk.run(handle, opts, rates=rates)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Release constants (checked against the repository's assets/release.json).
# ---------------------------------------------------------------------------

FRAME_SECONDS = 1920 / 48000          # YuE2-Vae downsampling_ratio 1920 @ 48 kHz -> 25 fps
CONTEXT = 24576                       # YuE2 protocol context length
KV_BYTES_PER_TOKEN = 2 * 28 * 8 * 128 * 2      # k+v * layers * kv_heads * head_dim * bf16
MOT_WEIGHT_BYTES = 7_261_441_640      # m-a-p/YuE2-3B model.safetensors, bf16
VAE_WEIGHT_BYTES = 530_512_720        # m-a-p/YuE2-Vae model.safetensors, fp32
DEFAULT_MODEL = "m-a-p/YuE2-3B"
DEFAULT_VAE = "m-a-p/YuE2-Vae"

# Runtime errors that mean "this GPU cannot use the CUDA-graph/flash fast path".
GRAPH_FALLBACK_HINTS = (
    "flash", "flashattention", "sm_80", "ampere", "compute capability",
    "cuda graph", "cudagraph",
)

DEFAULT_STYLE = (
    "English, warm piano pop, expressive female voice, acoustic piano, "
    "rounded bass and light drums, lyrical memorable melody, unhurried phrasing, 88 BPM"
)
DEFAULT_LYRICS = (
    "[Verse]\nNeon fades along the lane\nFootsteps keep the time of rain\n"
    "[Chorus]\nLet the day come into view\nEvery road begins with you"
)


def frames_for_seconds(seconds: float) -> int:
    return max(1, int(math.ceil(float(seconds) / FRAME_SECONDS)))


def seconds_for_frames(frames: int) -> float:
    return float(frames) * FRAME_SECONDS


def hms(seconds: float | None) -> str:
    if seconds is None or not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
        return "n/a"
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def find_local_models(search_roots=(Path("/kaggle/input"),)) -> tuple[str | None, str | None]:
    """Locate a YuE2 model/VAE already attached as a Kaggle dataset (offline use).

    Only three directory levels are probed so a large attached dataset cannot
    turn the preflight into a filesystem crawl.
    """
    model = vae = None
    for root in search_roots:
        if not root.is_dir():
            continue
        candidates = []
        for depth in (1, 2, 3):
            candidates.extend(sorted(root.glob("/".join(["*"] * depth) + "/config.json")))
        for config_path in candidates:
            directory = config_path.parent
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            kind = config.get("model_type")
            if kind == "yue2" and (directory / "qwen.tiktoken").is_file() and model is None:
                model = str(directory)
            elif kind == "yue2_vae" and (directory / "model.safetensors").is_file() and vae is None:
                vae = str(directory)
    return model, vae


def _disk(path: Path) -> dict:
    try:
        usage = shutil.disk_usage(path)
        return {"path": str(path), "free_gib": usage.free / 2**30, "total_gib": usage.total / 2**30}
    except OSError as exc:
        return {"path": str(path), "error": str(exc)}


def _ram_gib() -> float | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        return None


def _reachable(host: str, timeout: float = 4.0) -> bool:
    """TCP probe; a clone/pip failure saying 'Could not resolve host' usually
    means the notebook's Internet switch is off, not a broken network."""
    import socket

    try:
        socket.create_connection((host, 443), timeout=timeout).close()
        return True
    except OSError:
        return False


def _matmul_probe(torch, device, size=2048, repeats=3) -> dict:
    """Effective BF16/FP16/FP32 GEMM throughput; exposes emulated-BF16 penalties."""
    rates: dict = {}
    for name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32", torch.float32)):
        try:
            left = torch.randn(size, size, device=device, dtype=dtype)
            right = torch.randn(size, size, device=device, dtype=dtype)
            for _ in range(2):
                left @ right
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            for _ in range(repeats):
                left @ right
            torch.cuda.synchronize(device)
            rates[name] = 2 * size**3 / ((time.perf_counter() - start) / repeats) / 1e12
            del left, right
        except Exception as exc:                     # pragma: no cover - device dependent
            rates[name] = None
            rates[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rates


def environment(probe: bool = True) -> dict:
    """Collect everything needed to decide whether and how a run can proceed."""
    import torch

    report: dict = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "on_kaggle": Path("/kaggle").exists(),
        "ram_gib": _ram_gib(),
        "devices": [],
        "matmul_tflops": {},
    }
    if report["cuda_available"]:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            free, total = torch.cuda.mem_get_info(index)
            report["devices"].append({
                "index": index,
                "name": props.name,
                "memory_gib": total / 2**30,
                "free_gib": free / 2**30,
                "compute_capability": [props.major, props.minor],
                "multi_processor_count": props.multi_processor_count,
            })
        current = torch.cuda.current_device()
        report["current_device"] = f"cuda:{current}"
        try:
            report["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        except Exception as exc:                     # pragma: no cover - device dependent
            report["bf16_supported"] = None
            report["bf16_error"] = f"{type(exc).__name__}: {exc}"
        try:
            report["bf16_native"] = bool(torch.cuda.is_bf16_supported(including_emulation=False))
        except TypeError:                             # older torch without that argument
            report["bf16_native"] = None
        if probe:
            report["matmul_tflops"] = _matmul_probe(torch, torch.device(report["current_device"]))
    report["disk"] = {"working (/kaggle/working)": _disk(Path("/kaggle/working")),
                      "home cache (~/.cache)": _disk(Path.home()),
                      "root (/)": _disk(Path("/"))}
    report["local_model_dir"], report["local_vae_dir"] = find_local_models()
    report["internet"] = {"huggingface.co": _reachable("huggingface.co")}
    return report


def verdict(report: dict) -> dict:
    """Turn the environment report into a pass/fail plus concrete advice."""
    checks: list[str] = []
    warnings: list[str] = []
    if not report.get("cuda_available"):
        checks.append("No CUDA GPU. Kaggle: Settings -> Accelerator -> 'GPU T4 x2', then re-run.")
        return {"ok": False, "checks": checks, "warnings": warnings}
    index = int(str(report.get("current_device", "cuda:0")).split(":")[-1])
    device = report["devices"][index]
    report["device"] = device
    major, minor = device["compute_capability"]
    checks.append(f"{device['name']} (cc {major}.{minor}, {device['memory_gib']:.1f} GiB, "
                  f"{device['free_gib']:.1f} GiB free), torch {report['torch']}")
    if report.get("bf16_supported") is False:
        checks.append("FAIL: yue2's unquantized preset refuses this GPU because "
                      "torch.cuda.is_bf16_supported() is False. cc 6.0 (P100) is the usual "
                      "cause - switch the accelerator to 'GPU T4 x2'.")
        return {"ok": False, "checks": checks, "warnings": warnings}
    if report.get("bf16_native") is False:
        warnings.append("BF16 is emulated on this GPU (no native bf16 tensor cores): results are "
                        "correct, but expect several times the runtime of an Ampere-or-newer card.")
    if device["memory_gib"] < 15.0:
        warnings.append(f"{device['memory_gib']:.1f} GiB VRAM: keep songs short; offload_ar stays on.")
    if device["free_gib"] < device["memory_gib"] - 1.0:
        warnings.append(f"{device['free_gib']:.1f} GiB VRAM free of {device['memory_gib']:.1f} GiB "
                        "(another process holds the GPU); restart the session if that looks wrong.")
    if report.get("ram_gib") and report["ram_gib"] < 12:
        warnings.append(f"{report['ram_gib']:.0f} GiB system RAM: the 7.3 GB CPU copy of the model "
                        "plus the decoder may be tight.")
    working = report["disk"]["working (/kaggle/working)"]
    if "free_gib" in working and working["free_gib"] < 12:
        warnings.append(f"/kaggle/working has {working['free_gib']:.0f} GiB free; the ~7.8 GB of "
                        "model files plus artifacts need room (Kaggle persists at most 20 GB).")
    offline = (report.get("internet") or {}).get("huggingface.co") is False
    if offline and not report.get("local_model_dir"):
        checks.append("FAIL: huggingface.co is unreachable and no local model copy was found, so "
                      "the weights cannot be downloaded. Turn Settings -> Internet on, or attach "
                      "the models as a Kaggle dataset under /kaggle/input.")
        return {"ok": False, "checks": checks, "warnings": warnings}
    if offline:
        warnings.append("huggingface.co is unreachable; using the local model copy instead.")
    return {"ok": True, "checks": checks, "warnings": warnings}


def pipeline_settings(report: dict, *, ode_steps: int | None = None) -> dict:
    """Memory settings that fit the detected device instead of assuming 24 GB."""
    device = report.get("device") or report["devices"][0]
    vram = float(device["memory_gib"])
    settings = {
        "memory_budget_gib": max(4.0, min(24.0, vram - 0.5)),
        "offload_ar": vram < 15.5,
        "vae_core_frames": 512 if vram < 20 else 1024,
        "quantization": "none",          # FP8 needs cc >= 8.9; never available on the free tier
        "verify_hashes": True,
        "reserve_gib": max(0.0, vram - 2.0 - max(4.0, min(24.0, vram - 0.5))),
    }
    if ode_steps is not None:
        settings["ode_steps"] = int(ode_steps)
    return settings


def print_report(report: dict) -> dict:
    result = verdict(report)
    lines = ["=== YuE2 / Kaggle preflight ==="]
    lines += [f"  {line}" for line in result["checks"]]
    for name, value in (report.get("matmul_tflops") or {}).items():
        if isinstance(value, float):
            lines.append(f"  {name} GEMM: {value:.1f} TFLOP/s")
    if report.get("ram_gib"):
        lines.append(f"  system RAM: {report['ram_gib']:.1f} GiB")
    for label, disk in (report.get("disk") or {}).items():
        if "free_gib" in disk:
            lines.append(f"  disk {label}: {disk['free_gib']:.1f} GiB free")
    if report.get("local_model_dir"):
        lines.append(f"  local model (dataset) : {report['local_model_dir']}")
    if report.get("local_vae_dir"):
        lines.append(f"  local VAE (dataset)   : {report['local_vae_dir']}")
    internet = report.get("internet")
    if internet:
        state = "OK" if internet.get("huggingface.co") else "NG (Settings -> Internet を On に)"
        lines.append(f"  huggingface.co        : {state}")
    for line in result["warnings"]:
        lines.append(f"  WARNING: {line}")
    if result["ok"]:
        lines.append("  -> unquantized pipeline can run here; run measure() for a real ETA.")
    lines.append("=================================")
    print("\n".join(lines), flush=True)
    return result


# ---------------------------------------------------------------------------
# Options and pipeline construction
# ---------------------------------------------------------------------------


@dataclass
class Options:
    """Everything one generation needs; safe to reuse across stages and sessions."""

    outdir: Path = Path("/kaggle/working/outputs/yue2-song")
    style: str = DEFAULT_STYLE
    lyrics: str = DEFAULT_LYRICS
    cot: str = "full"                    # full | melody | off
    seed: int = 831001
    seconds: float = 120.0               # audio length; frames = seconds / 0.04
    abc: str | None = None               # optional pre-written score (skips ABC planning)
    plan_max_tokens: int | None = None   # cap the ABC stage; None keeps the release 4096.
                                         # A truncated score is a quality change: it is for
                                         # smoke tests, not for comparable results.
    ode_steps: int = 32                  # release default; lower only as an explicit tradeoff
    model: str | None = None             # None -> /kaggle/input copy, else the HF repo
    vae: str | None = None
    revision: str | None = None
    vae_revision: str | None = None
    backend: str = "auto"                # auto | torch | torch-eager | vllm
    quantization: str = "none"
    offload_ar: bool | None = None       # None -> decided from VRAM
    vae_core_frames: int | None = None
    memory_budget_gib: float | None = None
    verify_hashes: bool = True
    token: str | None = None
    progress: bool = True
    budget_minutes: float = 600.0        # wall-clock guard for the whole run
    safety_minutes: float = 10.0         # stop this long before the budget is gone
    extra_semantic_margin: int = 64      # decode headroom beyond the requested frames
    frames: int = field(init=False)

    def __post_init__(self):
        self.outdir = Path(self.outdir)
        self.seconds = float(self.seconds)
        self.frames = frames_for_seconds(self.seconds)
        if self.cot not in {"full", "melody", "off"}:
            raise ValueError("cot must be full, melody or off")
        if self.abc is not None and self.cot == "off":
            raise ValueError("a supplied score requires cot=full or cot=melody")

    def semantic_sampling(self) -> dict:
        """Cap semantic tokens at the requested length (memory and runaway guard)."""
        budget = int(min(CONTEXT - 1024, self.frames + self.extra_semantic_margin))
        sampling = {"max_tokens": budget}
        if budget < 200:                     # release default min_tokens is 200
            sampling["min_tokens"] = max(1, budget - 8)
        return sampling

    def to_dict(self) -> dict:
        data = asdict(self)
        data["outdir"] = str(self.outdir)
        data["abc"] = None if self.abc is None else f"<{len(self.abc)} chars>"
        return data


def resolve_models(options: Options, report: dict | None = None) -> tuple[str, str]:
    model = options.model or (report or {}).get("local_model_dir") or DEFAULT_MODEL
    vae = options.vae or (report or {}).get("local_vae_dir") or DEFAULT_VAE
    return model, vae


def build_pipeline(report: dict, options: Options, backend: str, *, generation_config=None):
    """Construct a pipeline with device-appropriate memory settings."""
    from yue2 import YuE2Pipeline

    settings = pipeline_settings(report, ode_steps=options.ode_steps)
    model, vae = resolve_models(options, report)
    kwargs = dict(
        device="cuda",
        memory_budget_gib=options.memory_budget_gib or settings["memory_budget_gib"],
        backend=backend,
        quantization=options.quantization,
        offload_ar=settings["offload_ar"] if options.offload_ar is None else options.offload_ar,
        vae_core_frames=options.vae_core_frames or settings["vae_core_frames"],
        verify_hashes=options.verify_hashes,
        progress=options.progress,
    )
    if generation_config is not None:
        kwargs["generation_config"] = generation_config
    return YuE2Pipeline.from_pretrained(
        model, vae=vae, revision=options.revision, vae_revision=options.vae_revision,
        token=options.token, **kwargs)


def ode_config(options: Options):
    """Release generation config with the requested solver step count."""
    from yue2.protocol import GenerationConfig

    base = GenerationConfig().to_dict()
    base["ode_steps"] = int(options.ode_steps)
    return GenerationConfig.from_dict(base)


class PipelineHandle:
    """Pipeline plus the ability to rebuild when the GPU cannot execute the
    CUDA-graph/flash fast path (common on Turing/T4 and on old drivers)."""

    def __init__(self, report: dict, options: Options):
        self.report = report
        self.options = options
        self.generation_config = ode_config(options)
        self.backend = "torch" if options.backend == "auto" else options.backend
        self.pipe = build_pipeline(report, options, self.backend,
                                   generation_config=self.generation_config)

    def retry_eager(self, exc: BaseException) -> bool:
        """Rebuild with the eager backend after a graph/flash refusal."""
        message = str(exc).lower()
        if self.options.backend != "auto" or self.backend != "torch":
            return False
        if not any(hint in message for hint in GRAPH_FALLBACK_HINTS):
            return False
        print(f"[kaggle] {type(exc).__name__}: {exc}\n"
              "[kaggle] retrying with backend='torch-eager' (no CUDA graphs, no flash attention).",
              flush=True)
        self.close()
        self.backend = "torch-eager"
        self.pipe = build_pipeline(self.report, self.options, self.backend,
                                   generation_config=self.generation_config)
        return True

    def close(self):
        if getattr(self, "pipe", None) is not None:
            try:
                self.pipe.close()
            except Exception:                        # pragma: no cover - best effort
                pass
        self.pipe = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def run_with_retry(handle: PipelineHandle, stage):
    """Run ``stage(pipe)``; if this GPU refuses the graph path, rebuild and retry once."""
    try:
        return stage(handle.pipe)
    except (RuntimeError, NotImplementedError) as exc:
        if not handle.retry_eager(exc):
            raise
        return stage(handle.pipe)


def open_pipeline(report: dict, options: Options, *, log=print) -> PipelineHandle:
    handle = PipelineHandle(report, options)
    settings = pipeline_settings(report)
    log(f"[kaggle] pipeline ready: backend={handle.backend}, "
        f"device={handle.pipe.device}, offload_ar={handle.pipe.offload_ar}, "
        f"vae_core_frames={handle.pipe.vae_core_frames}, "
        f"budget={handle.pipe.memory_budget_gib:.1f} GiB "
        f"(device limit {settings['memory_budget_gib']:.1f} GiB)")
    return handle


# ---------------------------------------------------------------------------
# Honest estimates: measure the real model on the real GPU
# ---------------------------------------------------------------------------


def _timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


def measure(handle: PipelineHandle, *, ar_tokens: int = 16, frames: int = 128,
            log=print) -> dict:
    """Short real generations that bound this GPU's throughput.

    Costs a couple of minutes and answers "will my song finish in this session?"
    before anything long is started. Release sampling is used except for the
    hand-capped token budgets and the single solver step used for measurement.
    """
    import numpy as np
    from yue2.pipeline import SemanticResult
    from yue2.protocol import GenerationConfig

    options = handle.options
    out: dict = {"frames_measured": frames, "ar_tokens": ar_tokens}

    def plan_stage(pipe):                       # the measurement never uses the real ABC budget
        kwargs = dict(style=options.style, lyrics=options.lyrics, cot=options.cot,
                      seed=options.seed)
        if options.abc:
            kwargs["abc"] = options.abc
        if options.cot == "off":
            return pipe.plan(**kwargs)
        return pipe.plan(abc_sampling={"max_tokens": max(ar_tokens, 8), "min_tokens": 4}, **kwargs)

    plan, seconds = _timed(lambda: run_with_retry(handle, plan_stage))
    out["plan_seconds"] = seconds
    out["plan_external"] = plan.abc is not None and options.cot != "off"
    out["plan_tokens"] = len(plan.abc_ids)
    # output_tps excludes the prefill and is the number that matters for a long song.
    plan_tps = (plan.timing or {}).get("output_tps")
    if not plan_tps and out["plan_tokens"] and seconds > 0.25 and not out["plan_external"]:
        plan_tps = out["plan_tokens"] / seconds
    out["plan_tps"] = plan_tps
    log(f"[kaggle] planning probe: {seconds:.1f}s, {out['plan_tokens']} score tokens "
        f"(external score: {out['plan_external']})")

    semantic, seconds = _timed(lambda: run_with_retry(
        handle, lambda p: p.generate_semantic(plan, sampling={"max_tokens": max(ar_tokens, 8),
                                                              "min_tokens": 4})))
    out["semantic_seconds"] = seconds
    out["semantic_tokens"] = len(semantic.tokens)
    out["semantic_tps"] = (((semantic.timing or {}).get("output_tps"))
                           or ((len(semantic.tokens) / seconds) if seconds else None))
    log(f"[kaggle] semantic probe: {len(semantic.tokens)} tokens in {seconds:.1f}s "
        f"({out['semantic_tps']:.2f} tok/s)")

    # Flow matching: two sizes at one solver step give the linear fit
    #   time(n) = fixed_per_chunk + n * per_frame_per_velocity * (2 * steps)
    base = handle.generation_config
    try:
        handle.pipe.generation_config = GenerationConfig.from_dict({**base.to_dict(), "ode_steps": 1})
        rng = np.random.default_rng(options.seed)
        times = {}
        for size in (frames, frames * 2):
            tokens = rng.integers(0, 32768, size=size).tolist()
            probe = SemanticResult(plan, tokens, {}, False)
            _, seconds = _timed(lambda s=probe: run_with_retry(handle, lambda p: p.synthesize(s)))
            times[size] = seconds
            log(f"[kaggle] flow-matching probe: {size} frames in {seconds:.1f}s")
        slope = (times[frames * 2] - times[frames]) / frames     # 2 velocity evals per frame
        out["nf_per_frame_seconds"] = max(0.0, slope / 2)
        out["nf_fixed_per_chunk_seconds"] = max(0.0, times[frames] - slope * frames)
        out["ode_steps"] = int(base.ode_steps)
    finally:
        if handle.pipe is not None:
            handle.pipe.generation_config = handle.generation_config

    latents = np.random.default_rng(0).standard_normal((64, 64)).astype(np.float32)
    _, seconds = _timed(lambda: run_with_retry(handle, lambda p: p.decode(latents)))
    out["vae_seconds_per_frame"] = seconds / 64
    log(f"[kaggle] decoder probe: {seconds:.1f}s for 64 frames "
        f"({out['vae_seconds_per_frame'] * 1000:.0f} ms/frame)")
    return out


def predict(measured: dict, options: Options, *, plan=None, abc_tokens: int | None = None) -> dict:
    """Turn measurements into a per-stage ETA for the requested song."""
    frames = options.frames
    ode_steps = int(options.ode_steps)
    if plan is not None:
        prefix_tokens = len(plan.prefix)
    else:
        prefix_tokens = 64 + round(len(options.style) / 3.5) + round(len(options.lyrics) / 2.5) \
            + (round(len(options.abc) / 3.0) if options.abc else 0)
    chunk_size = max(1, (CONTEXT - prefix_tokens - 3) // 2)
    chunks = max(1, math.ceil(frames / chunk_size))

    stages: dict = {"frames": frames, "audio_seconds": seconds_for_frames(frames),
                    "chunks": chunks, "chunk_size": chunk_size}

    if options.cot == "off" or options.abc is not None:
        stages["plan"] = 0.0
    else:
        tokens = abc_tokens if abc_tokens is not None else min(4096, max(256, round(frames * 0.8)))
        stages["plan_tokens_assumed"] = tokens
        # Fall back to the semantic decode rate when the planning probe was never
        # truncated: both stages are the same AR decoder.
        tps = measured.get("plan_tps") or measured.get("semantic_tps")
        stages["plan_tps_used"] = tps
        stages["plan"] = (tokens / tps) if tps else None

    stages["semantic"] = (frames / measured["semantic_tps"]) if measured.get("semantic_tps") else None
    per_frame = measured.get("nf_per_frame_seconds")
    stages["flow_matching"] = (None if per_frame is None else
                               chunks * measured.get("nf_fixed_per_chunk_seconds", 0.0)
                               + frames * per_frame * 2 * ode_steps)
    stages["decode"] = measured.get("vae_seconds_per_frame", 0.0) * frames

    parts = [stages[key] for key in ("plan", "semantic", "flow_matching", "decode")]
    stages["total"] = sum(parts) if all(isinstance(v, float) for v in parts) else None
    usable = max(0.0, (options.budget_minutes - options.safety_minutes) * 60)
    stages["deadline_seconds"] = usable
    stages["fits_budget"] = stages["total"] is not None and stages["total"] <= usable
    return stages


def print_estimate(stages: dict, options: Options) -> None:
    print("--- estimated cost on this device ------------------------------------")
    print(f"  requested audio : {stages['audio_seconds']:.0f}s -> {stages['frames']} latent frames "
          f"(25 fps), {stages['chunks']} flow-matching chunk(s)")
    for key in ("plan", "semantic", "flow_matching", "decode", "total"):
        note = ""
        if key == "plan" and options.cot != "off" and options.abc is None:
            note = (f" (assumes {stages.get('plan_tokens_assumed')} score tokens at "
                    f"{stages.get('plan_tps_used')} tok/s; release cap 4096)")
        if key == "flow_matching":
            note = f" ({options.ode_steps} midpoint steps = {2 * int(options.ode_steps)} velocity evals)"
        print(f"  {key:<15}: {hms(stages.get(key))}{note}")
    if stages.get("total") is not None:
        print(f"  deadline        : {hms(stages['deadline_seconds'])} usable of "
              f"{options.budget_minutes:.0f} min  -> fits: {stages['fits_budget']}")
        if not stages["fits_budget"]:
            print("  NOTE: the estimate exceeds the wall-clock budget. Lower --seconds, lower "
                  "ode_steps, or split the run across sessions (the stages resume).")
    print("---------------------------------------------------------------------", flush=True)


# ---------------------------------------------------------------------------
# Staged generation with resume
# ---------------------------------------------------------------------------


def _load_semantic(plan, path: Path, metadata: dict):
    """Rebuild a SemanticResult from cached tokens."""
    import numpy as np
    from yue2.pipeline import SemanticResult

    tokens = np.load(path, allow_pickle=False)
    if tokens.ndim != 1 or tokens.dtype.kind not in "iu":
        raise ValueError(f"unexpected cached token array in {path}")
    return SemanticResult(plan, [int(t) for t in tokens.tolist()],
                          metadata.get("timing", {}), bool(metadata.get("truncated", False)))


def _synthesize_checkpointed(pipe, semantic, outdir: Path, *, log=print, cancelled=None):
    """``YuE2Pipeline.synthesize`` with one cached file per original chunk.

    Calls the module-level helpers the pipeline itself calls (``yue2.nar``) so a
    long song survives a session timeout; falls back to the public method if the
    internal signature ever changes.
    """
    import numpy as np

    outdir.mkdir(parents=True, exist_ok=True)
    try:
        from yue2.nar import CachedNAR, song_chunks
        config = pipe.generation_config
        model = pipe._load_model(for_nar=True)          # the same call synthesize() makes
        chunks = song_chunks(semantic.plan.prefix, semantic.tokens,
                             semantic.plan.request.seed, config.context)
    except (ImportError, AttributeError, TypeError) as exc:
        log(f"[kaggle] checkpointed synthesis unavailable ({type(exc).__name__}: {exc}); "
            "using public synthesize() without per-chunk resume")
        return pipe.synthesize(semantic, cancelled=cancelled)

    parts = []
    for index, chunk in enumerate(chunks):
        path = outdir / f"latent_chunk_{index:04d}.npy"
        if path.is_file():
            parts.append(np.load(path))
            log(f"[kaggle] chunk {index + 1}/{len(chunks)} resumed from disk")
            continue
        if cancelled is not None and cancelled():
            raise InterruptedError("cancelled before a flow-matching chunk")
        engine = CachedNAR(model, chunk)
        try:
            state = {"done": 0}

            def progress(completed, total, state=state, index=index):
                if completed - state["done"] >= max(1, total // 10):
                    state["done"] = completed
                    log(f"[kaggle] chunk {index + 1}/{len(chunks)}: {completed}/{total} steps")

            latents = engine.solve(steps=int(pipe.generation_config.ode_steps),
                                   cancelled=cancelled, on_progress=progress)
        finally:
            engine.close()
        array = np.asarray(latents, dtype=np.float32)
        np.save(path, array)
        parts.append(array)
        log(f"[kaggle] chunk {index + 1}/{len(chunks)} done")
    return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]


def _stage_plan(pipe, options: Options, *, log=print, cancelled=None):
    """``plan()`` with the caller's request.

    Release sampling is untouched unless the caller explicitly capped the ABC
    budget with ``plan_max_tokens``.
    """
    kwargs = dict(style=options.style, lyrics=options.lyrics, cot=options.cot,
                  seed=options.seed, cancelled=cancelled)
    if options.abc:
        kwargs["abc"] = options.abc
    if options.plan_max_tokens and options.cot != "off":
        kwargs["abc_sampling"] = {"max_tokens": int(options.plan_max_tokens),
                                  "min_tokens": min(32, int(options.plan_max_tokens))}
    return pipe.plan(**kwargs)


def run(handle: PipelineHandle, options: Options, *, rates: dict | None = None,
        resume: bool = True, log=print) -> dict:
    """Execute the four stages, caching each one inside ``options.outdir``."""
    import numpy as np
    from yue2 import SymbolicPlan
    from yue2.pipeline import SongResult
    from yue2.storage import identity, write_json

    outdir = options.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    state_path = outdir / "kaggle_run.json"
    deadline = time.time() + max(60.0, (options.budget_minutes - options.safety_minutes) * 60)
    cancelled = lambda: time.time() > deadline          # noqa: E731 - wall-clock guard

    state: dict = {"options": options.to_dict(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "budget_minutes": options.budget_minutes, "stages": {}}
    if state_path.is_file():
        try:
            state["previous"] = json.loads(state_path.read_text(encoding="utf-8")).get("stages")
        except ValueError:
            pass
    if rates:
        state["measured"] = rates

    def note(stage, seconds, extra=None):
        entry = {"seconds": round(seconds, 2), "at": time.strftime("%H:%M:%S")}
        entry.update(extra or {})
        state["stages"][stage] = entry
        write_json(state_path, state)
        log(f"[kaggle] stage '{stage}' finished in {hms(seconds)}")

    result_path = outdir / "result.json"
    if resume and result_path.is_file():
        log(f"[kaggle] already complete: {result_path}")
        return {"status": "already-complete", "outdir": str(outdir)}

    pipe = handle.pipe

    def guarded(label, call):
        """Run a stage; a wall-clock stop is reported instead of raised so the
        notebook keeps the cached stages and the session stays usable."""
        try:
            return call()
        except InterruptedError as exc:
            state["interrupted"] = {"stage": label, "reason": str(exc),
                                    "at": time.strftime("%H:%M:%S")}
            write_json(state_path, state)
            log(f"[kaggle] wall-clock guard stopped '{label}' ({exc}); "
                "re-run to continue from the cached stages.")
            return None

    # ---- stage 1: symbolic plan -------------------------------------------------
    if resume and (outdir / "plan_manifest.json").is_file():
        plan = SymbolicPlan.load(outdir)
        log(f"[kaggle] stage 'plan' resumed from disk ({len(plan.abc_ids)} score tokens)")
    else:
        outcome = guarded("plan", lambda: _timed(lambda: run_with_retry(
            handle, lambda p: _stage_plan(p, options, log=log, cancelled=cancelled))))
        if outcome is None:
            return {"status": "interrupted", "stage": "plan", "outdir": str(outdir)}
        plan, seconds = outcome
        plan.save(outdir)
        note("plan", seconds, {"truncated": bool(plan.truncated), "score_tokens": len(plan.abc_ids)})
        if plan.truncated:
            log("[kaggle] WARNING: planning hit its token cap, so the score is incomplete. "
                "Shorten the song or supply your own ABC if the result sounds clipped.")

    # ---- stage 2: semantic tokens ----------------------------------------------
    semantic_file = outdir / "semantic.npy"
    if resume and semantic_file.is_file():
        semantic = _load_semantic(plan, semantic_file, state["stages"].get("semantic") or {})
        log(f"[kaggle] stage 'semantic' resumed from disk ({len(semantic.tokens)} tokens)")
    else:
        outcome = guarded("semantic", lambda: _timed(lambda: run_with_retry(
            handle, lambda p: p.generate_semantic(plan, sampling=options.semantic_sampling(),
                                                  cancelled=cancelled))))
        if outcome is None:
            return {"status": "interrupted", "stage": "semantic", "outdir": str(outdir)}
        semantic, seconds = outcome
        np.save(semantic_file, np.asarray(semantic.tokens, dtype=np.int32))
        note("semantic", seconds, {"tokens": len(semantic.tokens),
                                   "truncated": bool(semantic.truncated)})
        if semantic.truncated:
            log("[kaggle] WARNING: semantic generation hit its token cap; the song may end early.")

    # ---- stage 3: flow matching ------------------------------------------------
    latent_file = outdir / "latent.npy"
    if resume and latent_file.is_file():
        latents = np.load(latent_file)
        log(f"[kaggle] stage 'flow_matching' resumed from disk ({latents.shape[0]} frames)")
    else:
        outcome = guarded("flow_matching", lambda: _timed(lambda: run_with_retry(
            handle, lambda p: _synthesize_checkpointed(p, semantic, outdir, log=log,
                                                       cancelled=cancelled))))
        if outcome is None:
            return {"status": "interrupted", "stage": "flow_matching", "outdir": str(outdir)}
        latents, seconds = outcome
        latents = np.asarray(latents, dtype=np.float32)
        np.save(latent_file, latents)
        note("flow_matching", seconds, {"frames": int(latents.shape[0])})

    # ---- stage 4: decode and canonical artifacts -------------------------------
    if resume and (outdir / "audio.flac").is_file() and result_path.is_file():
        log("[kaggle] stage 'decode' already complete")
        return {"status": "already-complete", "outdir": str(outdir)}

    audio, seconds = _timed(lambda: run_with_retry(handle, lambda p: p.decode(latents)))
    request = semantic.plan.request
    config = pipe.effective_config(request, None, options.semantic_sampling())
    song = SongResult(audio, 48000, semantic, latents, config, pipe.weights,
                      {"kaggle_stages": state["stages"]},
                      identity({"request": request.to_dict(), "config": config,
                                "weights": pipe.weights}))
    # result.json records a hash for every file in the directory, so keep the
    # runner's own log out of the directory while that manifest is written.
    if state_path.is_file():
        state_path.unlink()
    try:
        receipt = song.save_artifacts(outdir)
    except BaseException:
        write_json(state_path, state)
        raise
    state["stages"]["decode"] = {"seconds": round(seconds, 2), "at": time.strftime("%H:%M:%S"),
                                 "audio_seconds": receipt.get("audio_seconds")}
    state["completed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state["truncated"] = song.truncated
    write_json(state_path, state)
    log(f"[kaggle] stage 'decode' finished in {hms(seconds)}")
    log(f"[kaggle] done -> {outdir / 'audio.flac'} ({receipt.get('audio_seconds', 0):.1f}s of audio)")
    return {"status": "complete", "outdir": str(outdir), "truncated": song.truncated,
            "receipt": receipt}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common(parser):
    parser.add_argument("--outdir", type=Path, default=Path("/kaggle/working/outputs/yue2-song"))
    parser.add_argument("--style", default=DEFAULT_STYLE)
    parser.add_argument("--lyrics", default=None)
    parser.add_argument("--lyrics-file", type=Path, default=None)
    parser.add_argument("--abc-file", type=Path, default=None)
    parser.add_argument("--cot", choices=("full", "melody", "off"), default="full")
    parser.add_argument("--seed", type=int, default=831001)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--ode-steps", type=int, default=32)
    parser.add_argument("--plan-max-tokens", type=int, default=None,
                        help="cap the ABC planning stage (smoke tests only; truncates the score)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--vae", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--vae-revision", default=None)
    parser.add_argument("--backend", choices=("auto", "torch", "torch-eager", "vllm"), default="auto")
    parser.add_argument("--offload-ar", dest="offload_ar", action="store_true", default=None)
    parser.add_argument("--no-offload-ar", dest="offload_ar", action="store_false")
    parser.add_argument("--budget-minutes", type=float, default=600.0)
    parser.add_argument("--safety-minutes", type=float, default=10.0)
    parser.add_argument("--quiet", action="store_true")


def _options(args) -> Options:
    lyrics = args.lyrics
    if args.lyrics_file:
        lyrics = args.lyrics_file.read_bytes().decode("utf-8")
    options = Options(outdir=args.outdir, style=args.style,
                      lyrics=lyrics if lyrics is not None else DEFAULT_LYRICS,
                      cot=args.cot, seed=args.seed, seconds=args.seconds,
                      ode_steps=args.ode_steps, plan_max_tokens=args.plan_max_tokens,
                      model=args.model, vae=args.vae,
                      revision=args.revision, vae_revision=args.vae_revision,
                      backend=args.backend, offload_ar=args.offload_ar,
                      budget_minutes=args.budget_minutes,
                      safety_minutes=args.safety_minutes, progress=not args.quiet)
    if args.abc_file:
        options.abc = args.abc_file.read_bytes().decode("utf-8")
    return options


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="preflight only: device, memory, verdict")
    check.add_argument("--json", type=Path, default=None)
    bench = sub.add_parser("bench", help="preflight, real micro-benchmark, ETA")
    _add_common(bench)
    run_cmd = sub.add_parser("run", help="staged, resumable song generation")
    _add_common(run_cmd)
    run_cmd.add_argument("--skip-bench", action="store_true")
    args = parser.parse_args(argv)

    try:
        report = environment()
    except ImportError as exc:
        print(f"torch is not importable in this kernel ({exc}).\n"
              "Run the install cell first (Kaggle images already ship a CUDA torch).")
        return 2
    if not print_report(report)["ok"]:
        return 2
    if args.command == "check":
        if getattr(args, "json", None):
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return 0

    options = _options(args)
    with open_pipeline(report, options) as handle:
        rates = None
        if args.command == "bench" or not args.skip_bench:
            try:
                rates = measure(handle)
                print_estimate(predict(rates, options), options)
            except Exception as exc:                 # pragma: no cover - device dependent
                print(f"[kaggle] benchmark failed ({type(exc).__name__}: {exc}); continuing")
        if args.command == "bench":
            return 0
        outcome = run(handle, options, rates=rates)
        print(json.dumps(outcome, indent=2, ensure_ascii=False, default=str)[:4000])
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
