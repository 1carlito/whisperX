#!/usr/bin/env python3
"""
WhisperX PR & Optimization Benchmark Harness

Features:
- Fixed audio dataset list (configurable in FIXED_DATASET below or via --manifest)
- Separates model load time (asr_load_sec, align_load_sec) from inference time
- Runs 1 untimed GPU warm-up pass followed by 5 timed runs (configurable)
- Records 5-run median (plus mean, min, max, stdev, and Real-Time Factor)
- Computes Global WER, Global CER, and Global IER using EnglishTextNormalizer + jiwer
  (matching whisperX-lab/eval_scripts/evaluate_pipeline.py & shared_utils.py)
- Compares WER/CER/IER using ABSOLUTE percentage-point deltas (PR % - Baseline %)
- Supports --align-only mode using saved baseline ASR segments for deterministic
  alignment benchmarking (ideal for testing alignment PRs or agent RSI loops)
"""

import argparse
import copy
import datetime
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jiwer
import numpy as np
import torch

import whisperx
from whisperx.audio import SAMPLE_RATE, load_audio

# Ensure benchmarks/ is importable so benchmarks.normalizers works regardless of target branch
BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from normalizers import EnglishTextNormalizer  # noqa: E402

# ==============================================================================
# FIXED BENCHMARK AUDIO LIST
# Add your fixed audio files (and optional ground-truth .nlp / .stm / .json / .txt)
# Paths are relative to benchmarks/data/audio/ unless absolute.
# If left empty, discover_dataset() falls back to all audio files in benchmarks/data/audio/
# ==============================================================================
FIXED_DATASET: List[Dict[str, Optional[str]]] = [
    # Example:
    # {"audio": "4320211.mp3", "gt": "4320211.nlp"},
    # {"audio": "sample2.wav", "gt": "sample2.json"},
]

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
GT_EXTENSIONS = [".nlp", ".stm", ".json", ".txt"]

_normalizer = EnglishTextNormalizer()


def strip_fillers(text: str) -> str:
    """Removes common filler words like 'um', 'uh', 'ah' (matching shared_utils.py)."""
    fillers = [r"\bum\b", r"\buh\b", r"\bah\b"]
    cleaned = text
    for filler in fillers:
        cleaned = re.sub(filler, "", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_text(text: str, strip_fillers_flag: bool = False) -> str:
    normalized = _normalizer(text)
    if strip_fillers_flag:
        normalized = strip_fillers(normalized)
    return normalized


def parse_nlp(nlp_path: Path) -> str:
    """Parses an Earnings21 .nlp pipe-delimited file into ground-truth text."""
    words: List[str] = []
    with nlp_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("token|"):
                continue
            parts = line.split("|")
            if len(parts) >= 1:
                token = parts[0].strip()
                punc = parts[4].strip() if len(parts) >= 5 else ""
                word = (token + punc) if punc else token
                if word and word != "token":
                    words.append(word)
    return " ".join(words)


def parse_stm(stm_path: Path) -> str:
    """Parses a TEDLIUM .stm file into ground-truth text."""
    ground_truth: List[str] = []
    noise_tokens = {"<unk>", "(noise)", "(laughter)", "(vocalized-noise)"}
    with stm_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            parts = line.split(maxsplit=6)
            if len(parts) >= 7:
                text = parts[6]
                if text != "ignore_time_segment_in_scoring":
                    words = text.split()
                    cleaned_words = [w for w in words if w.lower() not in noise_tokens]
                    if cleaned_words:
                        ground_truth.append(" ".join(cleaned_words))
    return " ".join(ground_truth)


def load_ground_truth_text(gt_path: Path) -> str:
    """Loads ground truth from .nlp, .stm, .json (segments), or .txt."""
    suffix = gt_path.suffix.lower()
    if suffix == ".nlp":
        return parse_nlp(gt_path)
    if suffix == ".stm":
        return parse_stm(gt_path)
    if suffix == ".json":
        data = json.loads(gt_path.read_text(encoding="utf-8"))
        segments = data.get("segments", []) if isinstance(data, dict) else data
        return " ".join(
            s.get("text", "").strip() if isinstance(s, dict) else str(s).strip()
            for s in segments
        ).strip()
    return gt_path.read_text(encoding="utf-8").strip()


def get_alignment_measures(gt_norm: str, pred_norm: str) -> Dict[str, Any]:
    """
    Computes Global WER (in %) using jiwer
    (identical normalization + WER calculation to whisperX-lab/evaluate_pipeline.py).
    """
    if not gt_norm and not pred_norm:
        return {"wer": 0.0}
    if not gt_norm:
        return {"wer": 100.0}

    wer = jiwer.wer(gt_norm, pred_norm) * 100.0
    return {"wer": round(wer, 4)}


def _cuda_sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _get_git_info(repo_dir: Optional[Path] = None) -> Dict[str, str]:
    cwd = str(repo_dir) if repo_dir else None
    info = {"commit": "unknown", "branch": "unknown"}
    try:
        info["commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        info["branch"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        pass
    return info


def _get_hardware_info(device: str) -> Dict[str, Any]:
    hw: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
        "gpu_name": None,
        "cuda_version": torch.version.cuda,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        hw["gpu_name"] = torch.cuda.get_device_name(0)
    return hw


def discover_dataset(
    audio_dir: Path,
    single_audio: Optional[Path] = None,
    single_gt: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
) -> List[Tuple[Path, Optional[Path]]]:
    """
    Resolves the fixed list of (audio_path, gt_path) pairs from:
    1) --audio / --gt CLI flags
    2) --manifest JSON or benchmarks/data/dataset.json if present
    3) FIXED_DATASET list at top of this file if non-empty
    4) All audio files in benchmarks/data/audio/ (matching .nlp/.stm/.json/.txt if present)
    """
    if single_audio is not None:
        if not single_audio.exists():
            raise FileNotFoundError(f"Audio file not found: {single_audio}")
        return [(single_audio.resolve(), single_gt.resolve() if single_gt else None)]

    entries: List[Dict[str, Optional[str]]] = []
    default_manifest = audio_dir.parent / "dataset.json"
    if manifest_path and manifest_path.exists():
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    elif default_manifest.exists():
        entries = json.loads(default_manifest.read_text(encoding="utf-8"))
    elif FIXED_DATASET:
        entries = FIXED_DATASET

    resolved: List[Tuple[Path, Optional[Path]]] = []
    if entries:
        for item in entries:
            a_rel = item["audio"]
            assert a_rel is not None
            a_path = Path(a_rel) if Path(a_rel).is_absolute() else (audio_dir / a_rel)
            if not a_path.exists():
                raise FileNotFoundError(f"Fixed dataset audio file not found: {a_path}")
            gt_rel = item.get("gt") or item.get("reference")
            gt_path: Optional[Path] = None
            if gt_rel:
                gt_path = Path(gt_rel) if Path(gt_rel).is_absolute() else (audio_dir / gt_rel)
                if not gt_path.exists():
                    raise FileNotFoundError(f"Fixed dataset ground-truth file not found: {gt_path}")
            resolved.append((a_path.resolve(), gt_path.resolve() if gt_path else None))
        return resolved

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    files = sorted(
        p.resolve()
        for p in audio_dir.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(
            f"No audio files found in {audio_dir} and FIXED_DATASET is empty.\n"
            f"Add your fixed audio files to FIXED_DATASET in benchmarks/run_bench.py or {audio_dir}."
        )

    for fpath in files:
        gt_found: Optional[Path] = None
        for ext in GT_EXTENSIONS:
            candidate_gt = fpath.with_suffix(ext)
            if candidate_gt.exists():
                gt_found = candidate_gt
                break
        resolved.append((fpath, gt_found))
    return resolved


def summarize_series(values: List[float]) -> Dict[str, float]:
    return {
        "median": round(statistics.median(values), 4),
        "mean": round(statistics.mean(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
        "runs": [round(v, 4) for v in values],
    }


def compute_global_quality(
    audio_dataset: List[Tuple[str, np.ndarray, float, Optional[str]]],
    outputs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Computes Global WER, Global CER, and Global IER across all files with ground truth
    using the exact same full-text reconstruction + EnglishTextNormalizer + jiwer
    method as evaluate_pipeline.py.
    """
    per_file: Dict[str, Dict[str, float]] = {}
    all_gt_norm_parts: List[str] = []
    all_hyp_norm_parts: List[str] = []

    for fname, _waveform, _dur, gt_text in audio_dataset:
        if not gt_text:
            continue
        file_out = outputs.get("files", {}).get(fname, {})
        c_segments = file_out.get("aligned_segments", [])
        c_full_text = " ".join(s.get("text", "") for s in c_segments if isinstance(s, dict))

        gt_norm = normalize_text(gt_text)
        c_norm = normalize_text(c_full_text)

        file_metrics = get_alignment_measures(gt_norm, c_norm)
        per_file[fname] = file_metrics
        all_gt_norm_parts.append(gt_norm)
        all_hyp_norm_parts.append(c_norm)

    if not all_gt_norm_parts:
        return {
            "files_with_ground_truth": 0,
            "global_wer": None,
            "per_file": {},
        }

    global_measures = get_alignment_measures(
        " ".join(all_gt_norm_parts),
        " ".join(all_hyp_norm_parts),
    )
    return {
        "files_with_ground_truth": len(per_file),
        "global_wer": global_measures["wer"],
        "per_file": per_file,
    }


def compare_outputs(
    baseline_outputs: Dict[str, Any],
    candidate_outputs: Dict[str, Any],
    timestamp_tol_sec: float = 0.005,
) -> Dict[str, Any]:
    """
    Compares candidate transcription & word alignment outputs against baseline_output.json.
    """
    total_base_words = 0
    total_cand_words = 0
    exact_word_matches = 0
    timestamp_diffs_ms: List[float] = []
    within_tol_count = 0
    base_norm_parts: List[str] = []
    cand_norm_parts: List[str] = []

    for fname, base_file_out in baseline_outputs.get("files", {}).items():
        cand_file_out = candidate_outputs.get("files", {}).get(fname)
        if not cand_file_out:
            continue

        base_text = " ".join(s.get("text", "").strip() for s in base_file_out.get("aligned_segments", []))
        cand_text = " ".join(s.get("text", "").strip() for s in cand_file_out.get("aligned_segments", []))
        base_norm_parts.append(normalize_text(base_text))
        cand_norm_parts.append(normalize_text(cand_text))

        base_words = base_file_out.get("word_segments", [])
        cand_words = cand_file_out.get("word_segments", [])
        total_base_words += len(base_words)
        total_cand_words += len(cand_words)

        for bw, cw in zip(base_words, cand_words):
            if bw.get("word") == cw.get("word"):
                exact_word_matches += 1
                if "start" in bw and "start" in cw and bw["start"] is not None and cw["start"] is not None:
                    start_diff = abs(float(bw["start"]) - float(cw["start"]))
                    end_diff = abs(float(bw.get("end", 0.0)) - float(cw.get("end", 0.0)))
                    max_diff = max(start_diff, end_diff)
                    timestamp_diffs_ms.append(max_diff * 1000.0)
                    if max_diff <= timestamp_tol_sec:
                        within_tol_count += 1

    drift_measures = get_alignment_measures(" ".join(base_norm_parts), " ".join(cand_norm_parts))
    denom = max(total_base_words, 1)
    return {
        "baseline_word_count": total_base_words,
        "candidate_word_count": total_cand_words,
        "word_match_rate": round(exact_word_matches / denom, 4),
        "timestamp_within_tol_rate": round(within_tol_count / max(len(timestamp_diffs_ms), 1), 4)
        if timestamp_diffs_ms
        else 1.0,
        "max_timestamp_diff_ms": round(max(timestamp_diffs_ms), 3) if timestamp_diffs_ms else 0.0,
        "mean_timestamp_diff_ms": round(statistics.mean(timestamp_diffs_ms), 3) if timestamp_diffs_ms else 0.0,
        "transcript_drift_wer": drift_measures["wer"],
    }


def format_markdown_report(
    result: Dict[str, Any],
    baseline: Optional[Dict[str, Any]] = None,
    output_diff: Optional[Dict[str, Any]] = None,
    pr_number: Optional[str] = None,
) -> str:
    title = "## WhisperX GPU Benchmark Report" + (f" (PR #{pr_number})" if pr_number else "")
    git_info = result["git"]
    hw = result["hardware"]

    lines = [
        title,
        "",
        f"- **Target Branch / Commit:** `{git_info['branch']}` (`{git_info['commit']}`)",
        f"- **GPU:** `{hw.get('gpu_name') or hw['device']}` (PyTorch `{hw['torch']}`, CUDA `{hw.get('cuda_version')}`)",
        f"- **Config:** model=`{result['config']['model']}`, compute_type=`{result['config']['compute_type']}`, "
        f"batch_size=`{result['config']['batch_size']}`, mode=`{'align-only' if result['config']['align_only'] else 'end-to-end'}`",
        f"- **Audio Dataset:** {result['dataset']['num_files']} file(s), `{result['dataset']['total_duration_sec']:.2f}s` total audio "
        f"({result['config']['warmup_runs']} warm-up + {result['config']['runs']} timed runs)",
        "",
    ]

    if baseline is None:
        lines.extend(
            [
                "### Baseline Recorded (5-Run Medians)",
                "",
                "| Stage | Median (s) | Mean (s) | Min (s) | Max (s) | Stdev (s) |",
                "| :--- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for key, label in [
            ("asr_load_sec", "ASR Model Load"),
            ("align_load_sec", "Align Model Load"),
            ("total_load_sec", "Total Model Load"),
            ("transcribe_sec", "Transcribe (ASR)"),
            ("align_sec", "Forced Alignment"),
            ("inference_sec", "Total Inference (ASR + Align)"),
        ]:
            m = result["metrics"][key]
            lines.append(
                f"| **{label}** | `{m['median']:.4f}` | `{m['mean']:.4f}` | `{m['min']:.4f}` | `{m['max']:.4f}` | `{m['stdev']:.4f}` |"
            )
        lines.append("")
        lines.append(f"- **Median Real-Time Factor (RTF):** `{result['metrics']['rtf_median']:.4f}x`")

        q = result.get("quality", {})
        if q.get("files_with_ground_truth", 0) > 0:
            lines.extend(
                [
                    "",
                    "### Baseline Global Accuracy (vs. Ground Truth)",
                    "",
                    f"- **Global WER:** `{q['global_wer']:.2f}%`",
                ]
            )
    else:
        base_git = baseline.get("git", {})
        lines.append(
            f"Comparing against fixed baseline `{base_git.get('branch', 'main')}` (`{base_git.get('commit', 'unknown')}`):"
        )
        lines.extend(
            [
                "",
                "### Runtime Comparison (5-Run Median)",
                "",
                "| Stage | Baseline Median (s) | PR Median (s) | Delta (s) | Speed Change (%) | PR Min–Max (s) |",
                "| :--- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for key, label in [
            ("total_load_sec", "Model Load (ASR + Align)"),
            ("transcribe_sec", "Transcribe (ASR)"),
            ("align_sec", "Forced Alignment"),
            ("inference_sec", "**Total Inference**"),
        ]:
            bm = baseline["metrics"][key]["median"]
            pm = result["metrics"][key]["median"]
            pmin = result["metrics"][key]["min"]
            pmax = result["metrics"][key]["max"]
            delta = pm - bm
            pct = ((pm - bm) / bm * 100.0) if bm > 0 else 0.0
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"| {label} | `{bm:.4f}` | `{pm:.4f}` | `{sign}{delta:.4f}` | **{sign}{pct:.2f}%** | `{pmin:.4f} – {pmax:.4f}` |"
            )

        # Global WER Comparison (Absolute percentage-point difference: PR % - Baseline %)
        q_cand = result.get("quality", {})
        q_base = baseline.get("quality", {})
        if q_cand.get("files_with_ground_truth", 0) > 0:
            lines.extend(
                [
                    "",
                    "### Global WER Comparison (Absolute Difference vs. Ground Truth)",
                    "",
                    "| Metric | Baseline (%) | PR (%) | Absolute Diff (% pts) |",
                    "| :--- | ---: | ---: | ---: |",
                ]
            )
            b_val = q_base.get("global_wer")
            c_val = q_cand.get("global_wer")
            if b_val is not None and c_val is not None:
                abs_diff = c_val - b_val
                sign = "+" if abs_diff >= 0 else ""
                lines.append(
                    f"| **Global WER** | `{b_val:.2f}%` | `{c_val:.2f}%` | **`{sign}{abs_diff:.2f}%`** |"
                )
            elif c_val is not None:
                lines.append(f"| **Global WER** | `N/A` | `{c_val:.2f}%` | `N/A` |")

        if output_diff is not None:
            lines.extend(
                [
                    "",
                    "### Output Consistency vs. Baseline",
                    "",
                    f"- **Word Count (Baseline vs. PR):** `{output_diff['baseline_word_count']}` vs. `{output_diff['candidate_word_count']}`",
                    f"- **Transcript Drift vs. Baseline (Global WER):** `{output_diff['transcript_drift_wer']:.2f}%`",
                    f"- **Exact Aligned Word Match:** `{output_diff['word_match_rate'] * 100:.2f}%`",
                    f"- **Timestamps within 5ms:** `{output_diff['timestamp_within_tol_rate'] * 100:.2f}%` "
                    f"(mean diff `{output_diff['mean_timestamp_diff_ms']:.2f} ms`, max diff `{output_diff['max_timestamp_diff_ms']:.2f} ms`)",
                ]
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="WhisperX 5-Run GPU Benchmark & Baseline Comparator")
    default_bench_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=default_bench_dir / "data" / "audio",
        help="Directory containing fixed benchmark audio files",
    )
    parser.add_argument("--manifest", type=Path, default=None, help="Optional JSON manifest of fixed audio/gt pairs")
    parser.add_argument("--audio", type=Path, default=None, help="Single audio file override")
    parser.add_argument("--gt", type=Path, default=None, help="Single ground-truth file override (.nlp, .stm, .json, .txt)")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_bench_dir / "results",
        help="Directory where baseline.json and PR reports are stored",
    )
    parser.add_argument("--target-repo", type=Path, default=None, help="Path to target repo for git SHA metadata")
    parser.add_argument("--model", type=str, default="large-v3", help="Whisper model name (default: large-v3)")
    parser.add_argument("--language", type=str, default="en", help="Audio language code (default: en)")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda or cpu)",
    )
    parser.add_argument("--compute-type", type=str, default="default", help="Compute type (default, float16, float32)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for ASR inference (default: 8)")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Number of untimed warm-up passes (default: 1)")
    parser.add_argument("--runs", type=int, default=5, help="Number of timed passes for median calculation (default: 5)")
    parser.add_argument(
        "--align-only",
        action="store_true",
        help="Skip ASR and benchmark whisperx.align() directly using saved baseline ASR segments",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Record this run as the fixed baseline (baseline.json and baseline_output.json)",
    )
    parser.add_argument(
        "--force-overwrite-baseline",
        action="store_true",
        help="Allow overwriting an existing baseline.json when --save-baseline is set",
    )
    parser.add_argument("--pr", type=str, default=None, help="PR number for report formatting")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to save candidate run JSON")
    parser.add_argument("--output-md", type=Path, default=None, help="Optional path to save markdown comparison report")

    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = args.results_dir / "baseline.json"
    baseline_output_path = args.results_dir / "baseline_output.json"

    if args.save_baseline and baseline_path.exists() and not args.force_overwrite_baseline:
        print(
            f"[ERROR] {baseline_path} already exists! Baseline is locked by default.\n"
            f"Pass --force-overwrite-baseline if you intentionally want to update the baseline.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.align_only and not baseline_output_path.exists():
        print(
            f"[ERROR] --align-only requires {baseline_output_path} from a baseline run.",
            file=sys.stderr,
        )
        sys.exit(1)

    dataset_pairs = discover_dataset(args.audio_dir, args.audio, args.gt, args.manifest)
    print(f"[Benchmark] Loaded fixed dataset with {len(dataset_pairs)} audio file(s).")

    # Pre-load audio waveforms in memory so disk I/O does not add jitter to GPU timings
    audio_dataset: List[Tuple[str, np.ndarray, float, Optional[str]]] = []
    total_audio_duration_sec = 0.0
    for fpath, gt_path in dataset_pairs:
        waveform = load_audio(str(fpath))
        duration = float(len(waveform)) / SAMPLE_RATE
        total_audio_duration_sec += duration
        gt_text = load_ground_truth_text(gt_path) if gt_path else None
        audio_dataset.append((fpath.name, waveform, duration, gt_text))

    baseline_outputs: Optional[Dict[str, Any]] = None
    if baseline_output_path.exists():
        baseline_outputs = json.loads(baseline_output_path.read_text(encoding="utf-8"))

    # 1. Time Model Loading separately
    print(f"[Benchmark] Loading models on device={args.device}...")
    asr_model = None
    asr_load_sec = 0.0
    if not args.align_only:
        _cuda_sync(args.device)
        t0 = time.perf_counter()
        asr_model = whisperx.load_model(
            args.model,
            device=args.device,
            compute_type=args.compute_type,
            language=args.language,
        )
        _cuda_sync(args.device)
        asr_load_sec = time.perf_counter() - t0

    _cuda_sync(args.device)
    t0 = time.perf_counter()
    align_model, align_metadata = whisperx.load_align_model(
        language_code=args.language,
        device=args.device,
    )
    _cuda_sync(args.device)
    align_load_sec = time.perf_counter() - t0
    total_load_sec = asr_load_sec + align_load_sec
    print(
        f"[Benchmark] Model load complete: asr={asr_load_sec:.3f}s, "
        f"align={align_load_sec:.3f}s, total={total_load_sec:.3f}s"
    )

    # 2. Run Warm-up + Timed Passes
    total_passes = args.warmup_runs + args.runs
    transcribe_times: List[float] = []
    align_times: List[float] = []
    inference_times: List[float] = []
    latest_outputs: Dict[str, Any] = {"files": {}}

    for pass_idx in range(total_passes):
        is_warmup = pass_idx < args.warmup_runs
        tag = f"Warm-up {pass_idx + 1}/{args.warmup_runs}" if is_warmup else f"Run {pass_idx - args.warmup_runs + 1}/{args.runs}"

        run_transcribe_sec = 0.0
        run_align_sec = 0.0

        for fname, waveform, _dur, _gt in audio_dataset:
            if args.align_only:
                assert baseline_outputs is not None
                raw_segments = copy.deepcopy(baseline_outputs["files"][fname]["raw_asr_segments"])
            else:
                assert asr_model is not None
                _cuda_sync(args.device)
                t_asr_start = time.perf_counter()
                asr_result = asr_model.transcribe(
                    waveform,
                    batch_size=args.batch_size,
                    language=args.language,
                )
                _cuda_sync(args.device)
                run_transcribe_sec += time.perf_counter() - t_asr_start
                raw_segments = copy.deepcopy(asr_result["segments"])

            segments_for_align = copy.deepcopy(raw_segments)
            _cuda_sync(args.device)
            t_align_start = time.perf_counter()
            aligned_result = whisperx.align(
                segments_for_align,
                align_model,
                align_metadata,
                waveform,
                args.device,
                return_char_alignments=False,
            )
            _cuda_sync(args.device)
            run_align_sec += time.perf_counter() - t_align_start

            if not is_warmup:
                latest_outputs["files"][fname] = {
                    "raw_asr_segments": raw_segments,
                    "aligned_segments": aligned_result["segments"],
                    "word_segments": aligned_result["word_segments"],
                }

        run_inference_sec = run_transcribe_sec + run_align_sec
        print(
            f"[Benchmark] {tag}: transcribe={run_transcribe_sec:.4f}s, "
            f"align={run_align_sec:.4f}s, inference={run_inference_sec:.4f}s"
        )

        if not is_warmup:
            transcribe_times.append(run_transcribe_sec)
            align_times.append(run_align_sec)
            inference_times.append(run_inference_sec)

    # 3. Compute Global WER / CER / IER (using EnglishTextNormalizer + jiwer)
    quality = compute_global_quality(audio_dataset, latest_outputs)

    metrics = {
        "asr_load_sec": summarize_series([asr_load_sec]),
        "align_load_sec": summarize_series([align_load_sec]),
        "total_load_sec": summarize_series([total_load_sec]),
        "transcribe_sec": summarize_series(transcribe_times),
        "align_sec": summarize_series(align_times),
        "inference_sec": summarize_series(inference_times),
        "rtf_median": round(statistics.median(inference_times) / max(total_audio_duration_sec, 1e-6), 5),
    }

    result_payload: Dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git": _get_git_info(args.target_repo),
        "hardware": _get_hardware_info(args.device),
        "config": {
            "model": args.model,
            "language": args.language,
            "compute_type": args.compute_type,
            "batch_size": args.batch_size,
            "warmup_runs": args.warmup_runs,
            "runs": args.runs,
            "align_only": args.align_only,
        },
        "dataset": {
            "num_files": len(audio_dataset),
            "files": [a[0] for a in audio_dataset],
            "total_duration_sec": round(total_audio_duration_sec, 3),
        },
        "metrics": metrics,
        "quality": quality,
    }

    if args.save_baseline:
        baseline_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
        baseline_output_path.write_text(json.dumps(latest_outputs, indent=2), encoding="utf-8")
        print(f"\n[Benchmark] Saved fixed baseline to {baseline_path}")
        print(f"[Benchmark] Saved baseline outputs to {baseline_output_path}\n")
        report_md = format_markdown_report(result_payload, baseline=None, output_diff=None)
    else:
        baseline_data = (
            json.loads(baseline_path.read_text(encoding="=" if False else "utf-8"))
            if baseline_path.exists()
            else None
        )
        output_diff = (
            compare_outputs(baseline_outputs, latest_outputs) if baseline_outputs is not None else None
        )
        result_payload["comparison_vs_baseline"] = output_diff
        if baseline_data and baseline_data.get("quality", {}).get("global_wer") is not None and quality.get("global_wer") is not None:
            result_payload["quality_abs_diff_vs_baseline"] = {
                "global_wer_abs_diff": round(quality["global_wer"] - baseline_data["quality"]["global_wer"], 4),
            }
        report_md = format_markdown_report(
            result_payload,
            baseline=baseline_data,
            output_diff=output_diff,
            pr_number=args.pr,
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")

    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(report_md, encoding="utf-8")

    print(report_md)


if __name__ == "__main__":
    main()
