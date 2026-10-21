#!/usr/bin/env python3
"""
WhisperX PR & Optimization Benchmark Harness

Features:
- Separates model load time (asr_load_sec, align_load_sec) from inference time
- Runs 1 untimed GPU warm-up pass followed by 5 timed runs (configurable)
- Records 5-run median (plus mean, min, max, stdev, and Real-Time Factor)
- Saves or compares against a fixed baseline (baseline.json + baseline_output.json)
- Supports --align-only mode using saved baseline ASR segments for deterministic
  alignment benchmarking (ideal for testing alignment PRs or agent RSI loops)
- Includes Phase 2 readiness: automatic WER & proper-noun accuracy if ground-truth
  .txt files or references.json are present alongside the audio files
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

import numpy as np
import torch

import whisperx
from whisperx.audio import SAMPLE_RATE, load_audio

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


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


def _normalize_words(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return [w for w in text.split() if w]


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER) using standard word-level Levenshtein distance."""
    ref_words = _normalize_words(reference)
    hyp_words = _normalize_words(hypothesis)
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    dp = list(range(len(hyp_words) + 1))
    for i, rw in enumerate(ref_words, start=1):
        prev = dp[0]
        dp[0] = i
        for j, hw in enumerate(hyp_words, start=1):
            temp = dp[j]
            cost = 0 if rw == hw else 1
            dp[j] = min(
                dp[j] + 1,       # deletion
                dp[j - 1] + 1,   # insertion
                prev + cost,     # substitution
            )
            prev = temp
    return dp[-1] / len(ref_words)


def compute_proper_noun_recall(reference: str, hypothesis: str) -> Optional[float]:
    """
    Estimate proper-noun recall from capitalized words in the reference transcript
    (excluding sentence-initial words). Returns None if no proper nouns are found.
    """
    tokens = re.findall(r"(?:^|\.\s+|\?\s+|\!\s+)([A-Z][a-zA-Z0-9_-]+)|(\b[A-Z][a-zA-Z0-9_-]+\b)", reference)
    proper_nouns = [t[1].lower() for t in tokens if t[1]]
    if not proper_nouns:
        return None
    hyp_set = set(_normalize_words(hypothesis))
    hits = sum(1 for pn in proper_nouns if pn in hyp_set)
    return hits / len(proper_nouns)


def discover_audio_files(audio_dir: Path, single_audio: Optional[Path] = None) -> List[Path]:
    if single_audio is not None:
        if not single_audio.exists():
            raise FileNotFoundError(f"Audio file not found: {single_audio}")
        return [single_audio.resolve()]

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    files = sorted(
        p.resolve()
        for p in audio_dir.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(
            f"No audio files ({', '.join(sorted(AUDIO_EXTENSIONS))}) found in {audio_dir}.\n"
            f"Place your fixed benchmark audio files in {audio_dir} before running."
        )
    return files


def summarize_series(values: List[float]) -> Dict[str, float]:
    return {
        "median": round(statistics.median(values), 4),
        "mean": round(statistics.mean(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
        "runs": [round(v, 4) for v in values],
    }


def compare_outputs(
    baseline_outputs: Dict[str, Any],
    candidate_outputs: Dict[str, Any],
    timestamp_tol_sec: float = 0.005,
) -> Dict[str, Any]:
    """
    Compare candidate transcription & word alignment outputs against baseline_output.json.
    """
    total_base_words = 0
    total_cand_words = 0
    exact_word_matches = 0
    timestamp_diffs_ms: List[float] = []
    within_tol_count = 0
    text_wer_Across_files: List[float] = []

    for fname, base_file_out in baseline_outputs.get("files", {}).items():
        cand_file_out = candidate_outputs.get("files", {}).get(fname)
        if not cand_file_out:
            continue

        base_text = " ".join(s.get("text", "").strip() for s in base_file_out.get("aligned_segments", []))
        cand_text = " ".join(s.get("text", "").strip() for s in cand_file_out.get("aligned_segments", []))
        text_wer_Across_files.append(compute_wer(base_text, cand_text))

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
        "transcript_drift_wer": round(statistics.mean(text_wer_Across_files), 5)
        if text_wer_Across_files
        else 0.0,
    }


def format_markdown_report(
    result: Dict[str, Any],
    baseline: Optional[Dict[str, Any]] = None,
    output_diff: Optional[Dict[str, Any]] = None,
    pr_number: Optional[str] = None,
) -> str:
    title = f"## WhisperX GPU Benchmark Report" + (f" (PR #{pr_number})" if pr_number else "")
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
                "| Stage | Baseline Median (s) | PR Median (s) | Delta (s) | Change (%) | PR Min–Max (s) |",
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

        if output_diff is not None:
            lines.extend(
                [
                    "",
                    "### Output Consistency vs. Baseline",
                    "",
                    f"- **Word Count (Baseline vs. PR):** `{output_diff['baseline_word_count']}` vs. `{output_diff['candidate_word_count']}`",
                    f"- **Transcript Drift (WER vs. Baseline):** `{output_diff['transcript_drift_wer'] * 100:.2f}%`",
                    f"- **Exact Aligned Word Match:** `{output_diff['word_match_rate'] * 100:.2f}%`",
                    f"- **Timestamps within 5ms:** `{output_diff['timestamp_within_tol_rate'] * 100:.2f}%` "
                    f"(mean diff `{output_diff['mean_timestamp_diff_ms']:.2f} ms`, max diff `{output_diff['max_timestamp_diff_ms']:.2f} ms`)",
                ]
            )

    if result.get("quality", {}).get("files_with_ground_truth", 0) > 0:
        q = result["quality"]
        lines.extend(
            [
                "",
                "### Phase 2 Accuracy Metrics (vs. Ground Truth)",
                "",
                f"- **Ground-Truth WER:** `{q['mean_wer'] * 100:.2f}%` (across {q['files_with_ground_truth']} file(s))",
            ]
        )
        if q.get("mean_proper_noun_recall") is not None:
            lines.append(f"- **Proper-Noun Recall:** `{q['mean_proper_noun_recall'] * 100:.2f}%`")

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
    parser.add_argument("--audio", type=Path, default=None, help="Single audio file override")
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

    audio_files = discover_audio_files(args.audio_dir, args.audio)
    print(f"[Benchmark] Found {len(audio_files)} audio file(s) in dataset.")

    # Pre-load audio waveforms in memory so disk I/O does not add jitter to GPU timings
    audio_dataset: List[Tuple[str, np.ndarray, float, Optional[str]]] = []
    total_audio_duration_sec = 0.0
    for fpath in audio_files:
        waveform = load_audio(str(fpath))
        duration = float(len(waveform)) / SAMPLE_RATE
        total_audio_duration_sec += duration
        gt_path = fpath.with_suffix(".txt")
        gt_text = gt_path.read_text(encoding="utf-8").strip() if gt_path.exists() else None
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

    # 3. Compute Phase 2 Quality Metrics (if ground-truth .txt files exist)
    wer_scores: List[float] = []
    pn_recalls: List[float] = []
    for fname, _waveform, _dur, gt_text in audio_dataset:
        if gt_text:
            hyp_segments = latest_outputs["files"][fname]["aligned_segments"]
            hyp_text = " ".join(s.get("text", "").strip() for s in hyp_segments)
            wer_scores.append(compute_wer(gt_text, hyp_text))
            pn = compute_proper_noun_recall(gt_text, hyp_text)
            if pn is not None:
                pn_recalls.append(pn)

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
        "quality": {
            "files_with_ground_truth": len(wer_scores),
            "mean_wer": round(statistics.mean(wer_scores), 5) if wer_scores else None,
            "mean_proper_noun_recall": round(statistics.mean(pn_recalls), 5) if pn_recalls else None,
        },
    }

    if args.save_baseline:
        baseline_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
        baseline_output_path.write_text(json.dumps(latest_outputs, indent=2), encoding="utf-8")
        print(f"\n[Benchmark] Saved fixed baseline to {baseline_path}")
        print(f"[Benchmark] Saved baseline outputs to {baseline_output_path}\n")
        report_md = format_markdown_report(result_payload, baseline=None, output_diff=None)
    else:
        baseline_data = (
            json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else None
        )
        output_diff = (
            compare_outputs(baseline_outputs, latest_outputs) if baseline_outputs is not None else None
        )
        result_payload["comparison_vs_baseline"] = output_diff
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
