# WhisperX GPU Benchmark & PR Verification Harness

This harness implements the 3-phase WhisperX benchmarking system on your AWS GPU server (and also serves as the evaluation judge for agent optimization / RSI loops).

---

## 1. One-Time Setup on Your AWS GPU Server

Use **two separate clones** so switching branches/PRs in the target repo never touches your benchmark scripts, fixed audio files, or locked `baseline.json`:

```bash
# 1. Clone the benchmark harness (stays on test-env branch permanently)
git clone -b test-env https://github.com/1carlito/whisperX.git ~/whisperx-bench

# 2. Clone the target repo under test (switches between main and PRs)
git clone https://github.com/m-bain/whisperX.git ~/whisperx-target

# 3. Put your fixed small audio dataset in ~/whisperx-bench/benchmarks/data/audio/
cp /path/to/your/sample_audio.wav ~/whisperx-bench/benchmarks/data/audio/
# (Optional Phase 2): Add matching sample_audio.txt with ground-truth transcript for WER
```

Lock your HuggingFace/Torch cache directory in `~/.bashrc` so model downloads never skew timings:
```bash
export HF_HOME="$HOME/.cache/huggingface"
export TORCH_HOME="$HOME/.cache/torch"
```

---

## 2. Phase 1: Establish Your Fixed Baseline on `main`

Run the 1 warm-up + 5 timed passes on `main` and lock the median runtime and baseline outputs:

```bash
cd ~/whisperx-bench
bash benchmarks/compare_pr.sh baseline
```

- Saves `benchmarks/results/baseline.json` (5-run median `total_load_sec`, `transcribe_sec`, `align_sec`, `inference_sec`, RTF, commit SHA, GPU info).
- Saves `benchmarks/results/baseline_output.json` (raw ASR segments + aligned word timestamps).
- **Locked by default:** Running `baseline` again will refuse to overwrite `baseline.json` unless you explicitly pass `--force`.

---

## 3. Benchmarking a Pull Request

To test any PR (e.g., PR `#1481`) against the locked baseline:

```bash
cd ~/whisperx-bench

# Run locally and print the markdown comparison table:
bash benchmarks/compare_pr.sh pr 1481

# Or run AND automatically post the result table as a comment on PR #1481:
bash benchmarks/compare_pr.sh pr 1481 --post-comment
```

### Alignment-Only Mode (For Alignment PRs & Agent RSI Loops)
When testing alignment optimizations (such as PR `#1481` Numba JIT or `torchaudio.functional.forced_align` CUDA kernels), pass `--align-only` to feed the exact saved `main` ASR segments into `whisperx.align()`:

```bash
# Test a PR's alignment speed & timestamp equivalence:
bash benchmarks/compare_pr.sh pr 1481 --align-only

# Test a local branch/commit in ~/whisperx-target during an agent optimization loop:
bash benchmarks/compare_pr.sh ref my-cuda-align-branch --align-only
```

---

## 4. Phase 2 & Phase 3 Readiness

- **Phase 2 (WER & Proper-Noun Accuracy):** Drop `<audio_name>.txt` ground-truth files next to `<audio_name>.wav` in `benchmarks/data/audio/`. `run_bench.py` automatically detects them and adds a **Phase 2 Accuracy Metrics (WER & Proper-Noun Recall)** section to the report.
- **Phase 3 (GitHub Actions Manual Toggle):** `.github/workflows/benchmark_pr.yml` is pre-configured with two safe manual gates for a self-hosted GPU runner (`runs-on: [self-hosted, gpu]`):
  1. **Apply the `run-benchmark` label** to any PR on GitHub after inspecting its code.
  2. Or click **Actions → GPU PR Benchmark → Run workflow** and type the PR number.
