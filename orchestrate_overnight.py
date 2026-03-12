#!/usr/bin/env python3
"""
Overnight orchestrator: waits for fold 1 (already running), then runs
folds 2-8 sequentially, stopping before any new fold if time >= 07:30.
Writes a full recap to run_log.txt when done.
"""
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT      = Path("/mnt/c/Users/lhdee/OneDrive/Desktop/Project_Apex")
LOG_PATH  = ROOT / "logs" / "apex_run.log"
RUN_LOG   = ROOT / "run_log.txt"
PYTHON    = sys.executable
DEADLINE  = datetime.time(7, 30, 0)

ENV = {
    "PATH": "/home/apexproject/.local/bin:/usr/local/sbin:/usr/local/bin"
            ":/usr/sbin:/usr/bin:/sbin:/bin",
    "LD_LIBRARY_PATH": "/usr/lib/wsl/lib",
    "HOME": "/home/apexproject",
    "USER": "apexproject",
    "PYTHONUNBUFFERED": "1",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def ts():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"[{ts()}] {msg}", flush=True)

def is_past_deadline():
    return datetime.datetime.now().time() >= DEADLINE

def log_content_from_offset(byte_offset: int) -> str:
    """Return log content that was written after byte_offset."""
    try:
        with open(LOG_PATH, "rb") as f:
            f.seek(byte_offset)
            return f.read().decode("utf-8", errors="replace")
    except Exception:
        return ""

def wait_for_fold_complete(fold_id: int, byte_offset: int,
                           timeout_sec: int = 7200) -> bool:
    """Poll apex_run.log for 'FOLD {fold_id} COMPLETE' after byte_offset."""
    marker = f"FOLD {fold_id} COMPLETE"
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        new_text = log_content_from_offset(byte_offset)
        if marker in new_text:
            for line in new_text.splitlines():
                if marker in line:
                    log(f"Fold {fold_id} complete → {line.strip()}")
                    return True
        elapsed = int(time.time() - t0)
        log(f"  Still waiting for fold {fold_id} ... {elapsed}s elapsed")
        time.sleep(30)
    log(f"  TIMEOUT waiting for fold {fold_id} after {timeout_sec}s")
    return False

def read_fold_metrics(fold_id: int) -> dict:
    p = ROOT / "results" / f"fold_{fold_id}" / "oos_metrics.json"
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

def get_complete_log_line(fold_id: int, byte_offset: int) -> str:
    marker = f"FOLD {fold_id} COMPLETE"
    text = log_content_from_offset(byte_offset)
    for line in reversed(text.splitlines()):
        if marker in line:
            return line.strip()
    return ""

def current_log_size() -> int:
    try:
        return LOG_PATH.stat().st_size
    except Exception:
        return 0

def run_fold(fold_id: int):
    """Blocking: run one fold. Returns (success, elapsed_sec, byte_offset_before)."""
    byte_before = current_log_size()
    log(f"Starting fold {fold_id} ...")
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, "run_full_apex.py", "--fold", str(fold_id)],
        cwd=str(ROOT),
        env=ENV,
    )
    elapsed = time.time() - t0
    success = result.returncode == 0
    log(f"Fold {fold_id} {'OK' if success else 'FAILED (rc=' + str(result.returncode) + ')'}  "
        f"({elapsed/60:.1f} min)")
    return success, elapsed, byte_before


# ── recap writer ──────────────────────────────────────────────────────────────

def write_recap(completed: dict, errors: list):
    now = ts()
    lines = []
    lines.append("=" * 72)
    lines.append("PROJECT APEX — OVERNIGHT RUN RECAP")
    lines.append(f"Written: {now}")
    lines.append("=" * 72)
    lines.append("")

    # Per-fold table
    lines.append(f"{'Fold':<6} {'Status':<8} {'CAGR':>10} {'Sharpe':>8} "
                 f"{'Sortino':>10} {'MaxDD':>8} {'Elapsed':>8}")
    lines.append("-" * 64)
    for fid in range(1, 9):
        if fid not in completed:
            lines.append(f"{fid:<6} {'not run':<8}")
            continue
        info = completed[fid]
        status = "OK" if info["success"] else "FAILED"
        m = info.get("metrics", {})
        if m:
            cagr    = m.get("excess_cagr", float("nan"))
            sharpe  = m.get("sharpe", float("nan"))
            sortino = m.get("sortino", float("nan"))
            maxdd   = m.get("max_drawdown", float("nan"))
            elap    = info.get("elapsed_min", float("nan"))
            lines.append(
                f"{fid:<6} {status:<8} {cagr:>+10.4f} {sharpe:>8.4f} "
                f"{sortino:>10.4f} {maxdd:>8.4f} {elap:>7.1f}m"
            )
        else:
            lines.append(f"{fid:<6} {status:<8} (no metrics)")
    lines.append("")

    # Errors
    if errors:
        lines.append("ERRORS / ISSUES:")
        for e in errors:
            lines.append(f"  - {e}")
        lines.append("")

    # Cross-fold patterns
    ok_metrics = [
        completed[fid]["metrics"]
        for fid in sorted(completed)
        if completed[fid].get("success") and completed[fid].get("metrics")
    ]
    if len(ok_metrics) >= 2:
        import math
        def _mean(vals):
            clean = [v for v in vals if not math.isnan(v)]
            return sum(clean) / len(clean) if clean else float("nan")

        cagrs   = [m.get("excess_cagr", float("nan")) for m in ok_metrics]
        sharpes = [m.get("sharpe", float("nan")) for m in ok_metrics]
        maxdds  = [m.get("max_drawdown", float("nan")) for m in ok_metrics]
        cost_drags = [m.get("cost_drag", float("nan")) for m in ok_metrics]

        lines.append("CROSS-FOLD PATTERNS:")
        lines.append(f"  Avg excess CAGR : {_mean(cagrs):+.4f}")
        lines.append(f"  Avg Sharpe      : {_mean(sharpes):.4f}")
        lines.append(f"  Avg Sortino     : {_mean([m.get('sortino', float('nan')) for m in ok_metrics]):.2f}")
        lines.append(f"  Best MaxDD      : {max(maxdds):.4f}")
        lines.append(f"  Worst MaxDD     : {min(maxdds):.4f}")
        lines.append(f"  Avg cost_drag   : {_mean(cost_drags):.6f}")
        # Observations
        if any(abs(c) > 50 for c in cagrs if not math.isnan(c)):
            lines.append("  NOTE: One or more folds show extreme CAGR — may indicate")
            lines.append("        scale/normalization issues in the reward. Inspect closely.")
        ruin_folds = [fid for fid in sorted(completed)
                      if completed[fid].get("metrics", {}).get("max_drawdown", 0) <= -0.95]
        if ruin_folds:
            lines.append(f"  WARNING: Fold(s) {ruin_folds} reached near-total-loss (MaxDD <= -0.95)")
        lines.append("")

    # Next step
    next_fold = None
    for fid in range(1, 9):
        if fid not in completed or not completed[fid].get("success"):
            next_fold = fid
            break

    if next_fold:
        lines.append(f"NEXT STEP: Start fold {next_fold} (run: python run_full_apex.py --fold {next_fold})")
    else:
        lines.append("NEXT STEP: All 8 folds complete — ready for cross-fold analysis / report generation.")
    lines.append("")

    text = "\n".join(lines) + "\n"
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(text)
    log(f"Recap written to {RUN_LOG}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    log("Overnight orchestrator started.")
    log(f"Deadline: 07:30  |  Will stop before starting any new fold after that time.")

    completed = {}
    errors    = []

    # ── Fold 1: already running externally; wait for it ──────────────────────
    # Use a byte offset captured AFTER fold 1 was launched so we don't
    # accidentally match the "FOLD 1 COMPLETE" line from a prior run.
    log("Fold 1 is already running (launched by Claude Code). Waiting for completion...")
    byte_offset_fold1 = 824601  # log size at the moment fold 1 restart was confirmed live
    fold1_ok = wait_for_fold_complete(1, byte_offset_fold1, timeout_sec=7200)

    if fold1_ok:
        metrics = read_fold_metrics(1)
        log_line = get_complete_log_line(1, byte_offset_fold1)
        completed[1] = {
            "success": True,
            "metrics": metrics,
            "elapsed_min": float("nan"),  # not timed here
        }
    else:
        errors.append("Fold 1: timed out waiting (>2 h) — skipping and continuing")
        completed[1] = {"success": False, "metrics": {}}

    # ── Folds 2-8 ────────────────────────────────────────────────────────────
    for fold_id in range(2, 9):
        if is_past_deadline():
            log(f"Deadline 07:30 reached — stopping before fold {fold_id}.")
            errors.append(f"Folds {fold_id}-8: not started — deadline 07:30 reached.")
            break

        # First attempt
        success, elapsed, byte_before = run_fold(fold_id)

        if not success:
            log(f"Fold {fold_id} failed on first attempt. Retrying once...")
            errors.append(f"Fold {fold_id}: first attempt failed (rc != 0); retrying.")
            success, elapsed, byte_before = run_fold(fold_id)
            if not success:
                log(f"Fold {fold_id} failed on second attempt. Skipping.")
                errors.append(f"Fold {fold_id}: second attempt also failed — skipped.")
                completed[fold_id] = {"success": False, "metrics": {}, "elapsed_min": elapsed / 60}
                continue

        metrics  = read_fold_metrics(fold_id)
        log_line = get_complete_log_line(fold_id, byte_before)
        completed[fold_id] = {
            "success": True,
            "metrics": metrics,
            "elapsed_min": elapsed / 60,
        }
        log(f"Fold {fold_id} recorded. Metrics: {metrics}")

    # ── Write recap ──────────────────────────────────────────────────────────
    write_recap(completed, errors)
    log("Orchestrator done. Waiting for user instructions.")


if __name__ == "__main__":
    main()
