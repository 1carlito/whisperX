#!/usr/bin/env bash
set -euo pipefail

# WhisperX Remote GPU Server Runner
# Usage:
#   1) Record fixed baseline from main:
#      ./benchmarks/compare_pr.sh baseline [--force]
#
#   2) Benchmark a PR against the locked baseline:
#      ./benchmarks/compare_pr.sh pr <PR_NUMBER> [--post-comment] [--align-only]
#
#   3) Benchmark an arbitrary branch/commit in target clone (e.g. for agent RSI loops):
#      ./benchmarks/compare_pr.sh ref <GIT_REF> [--align-only]

BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_REPO="${WHISPERX_TARGET_REPO:-$HOME/whisperx-target}"
UPSTREAM_REMOTE="${WHISPERX_UPSTREAM_REMOTE:-upstream}"
MODEL="${WHISPERX_BENCH_MODEL:-large-v3}"
BATCH_SIZE="${WHISPERX_BENCH_BATCH_SIZE:-8}"
RUNS="${WHISPERX_BENCH_RUNS:-5}"
WARMUP="${WHISPERX_BENCH_WARMUP:-1}"

if [[ ! -d "$TARGET_REPO/.git" ]]; then
  echo "[ERROR] Target repo not found at $TARGET_REPO"
  echo "Clone a clean copy of whisperX there first:"
  echo "  git clone https://github.com/m-bain/whisperX.git \"$TARGET_REPO\""
  exit 1
fi

install_target() {
  echo "[Runner] Installing target whisperX from $TARGET_REPO (plus jiwer, more-itertools, regex)..."
  if command -v uv >/dev/null 2>&1; then
    uv pip install -e "$TARGET_REPO" jiwer more-itertools regex --quiet
  else
    pip install -e "$TARGET_REPO" jiwer more-itertools regex --quiet
  fi
}

MODE="${1:-}"
shift || true

case "$MODE" in
  baseline)
    FORCE_FLAG=""
    if [[ "${1:-}" == "--force" ]]; then
      FORCE_FLAG="--force-overwrite-baseline"
    fi
    echo "[Runner] Checking out main in $TARGET_REPO..."
    git -C "$TARGET_REPO" fetch origin main
    git -C "$TARGET_REPO" checkout -f FETCH_HEAD
    install_target

    python3 "$BENCH_ROOT/benchmarks/run_bench.py" \
      --target-repo "$TARGET_REPO" \
      --model "$MODEL" \
      --batch-size "$BATCH_SIZE" \
      --warmup-runs "$WARMUP" \
      --runs "$RUNS" \
      --save-baseline \
      $FORCE_FLAG
    ;;

  pr)
    PR_NUMBER="${1:-}"
    if [[ -z "$PR_NUMBER" ]]; then
      echo "Usage: $0 pr <PR_NUMBER> [--post-comment] [--align-only]"
      exit 1
    fi
    shift || true

    POST_COMMENT=false
    EXTRA_ARGS=()
    for arg in "$@"; do
      if [[ "$arg" == "--post-comment" ]]; then
        POST_COMMENT=true
      else
        EXTRA_ARGS+=("$arg")
      fi
    done

    echo "[Runner] Fetching PR #$PR_NUMBER in $TARGET_REPO..."
    REMOTE_NAME="origin"
    if git -C "$TARGET_REPO" remote get-url "$UPSTREAM_REMOTE" >/dev/null 2>&1; then
      REMOTE_NAME="$UPSTREAM_REMOTE"
    fi
    git -C "$TARGET_REPO" fetch "$REMOTE_NAME" "pull/$PR_NUMBER/head:pr-$PR_NUMBER" -f
    git -C "$TARGET_REPO" checkout -f "pr-$PR_NUMBER"
    install_target

    REPORT_MD="$BENCH_ROOT/benchmarks/results/pr_${PR_NUMBER}_report.md"
    REPORT_JSON="$BENCH_ROOT/benchmarks/results/pr_${PR_NUMBER}_result.json"

    python3 "$BENCH_ROOT/benchmarks/run_bench.py" \
      --target-repo "$TARGET_REPO" \
      --model "$MODEL" \
      --batch-size "$BATCH_SIZE" \
      --warmup-runs "$WARMUP" \
      --runs "$RUNS" \
      --pr "$PR_NUMBER" \
      --output-md "$REPORT_MD" \
      --output-json "$REPORT_JSON" \
      "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

    if [[ "$POST_COMMENT" == "true" ]]; then
      echo "[Runner] Posting benchmark report to PR #$PR_NUMBER..."
      gh pr comment "$PR_NUMBER" --body-file "$REPORT_MD"
    fi
    ;;

  ref)
    GIT_REF="${1:-}"
    if [[ -z "$GIT_REF" ]]; then
      echo "Usage: $0 ref <GIT_REF> [--align-only]"
      exit 1
    fi
    shift || true

    echo "[Runner] Checking out $GIT_REF in $TARGET_REPO..."
    git -C "$TARGET_REPO" checkout -f "$GIT_REF"
    install_target

    SAFE_REF="$(echo "$GIT_REF" | tr '/:' '__')"
    REPORT_MD="$BENCH_ROOT/benchmarks/results/ref_${SAFE_REF}_report.md"
    REPORT_JSON="$BENCH_ROOT/benchmarks/results/ref_${SAFE_REF}_result.json"

    python3 "$BENCH_ROOT/benchmarks/run_bench.py" \
      --target-repo "$TARGET_REPO" \
      --model "$MODEL" \
      --batch-size "$BATCH_SIZE" \
      --warmup-runs "$WARMUP" \
      --runs "$RUNS" \
      --output-md "$REPORT_MD" \
      --output-json "$REPORT_JSON" \
      "$@"
    ;;

  *)
    echo "Usage:"
    echo "  $0 baseline [--force]"
    echo "  $0 pr <PR_NUMBER> [--post-comment] [--align-only]"
    echo "  $0 ref <GIT_REF> [--align-only]"
    exit 1
    ;;
esac
