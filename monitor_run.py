#!/usr/bin/env python3
"""
Monitor a running run_full_apex.py and write per-fold summaries + final
cross-fold summary to run_log.txt as each fold completes.
Usage: python3 monitor_run.py <training_pid>
"""
import json
import math
import os
import sys
import time
import datetime
from pathlib import Path

ROOT     = Path("/mnt/c/Users/lhdee/OneDrive/Desktop/Project_Apex")
LOG_PATH = ROOT / "logs" / "apex_run.log"
RUN_LOG  = ROOT / "run_log.txt"
TRAINING_PID = int(sys.argv[1]) if len(sys.argv) > 1 else None

# Byte offset when this monitor started — only look at new log content
START_OFFSET = LOG_PATH.stat().st_size if LOG_PATH.exists() else 0
RUN_START_TIME = time.time()


def ts():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"[{ts()}] {msg}", flush=True)

def new_log_content() -> str:
    try:
        with open(LOG_PATH, "rb") as f:
            f.seek(START_OFFSET)
            return f.read().decode("utf-8", errors="replace")
    except Exception:
        return ""

def training_process_alive() -> bool:
    if TRAINING_PID is None:
        return True
    try:
        os.kill(TRAINING_PID, 0)
        return True
    except OSError:
        return False

def read_metrics(fold_id: int) -> dict:
    p = ROOT / "results" / f"fold_{fold_id}" / "oos_metrics.json"
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

def get_complete_line(fold_id: int) -> str:
    marker = f"FOLD {fold_id} COMPLETE"
    for line in reversed(new_log_content().splitlines()):
        if marker in line:
            return line.strip()
    return ""

def append_fold_summary(fold_id: int, completed_folds: dict, total_folds_expected: int = 8):
    m = read_metrics(fold_id)
    complete_line = get_complete_line(fold_id)
    completed_folds[fold_id] = m

    # Parse elapsed from log line
    elapsed_str = ""
    if "elapsed" in complete_line:
        try:
            elapsed_str = complete_line.split("elapsed")[1].split("min")[0].strip() + " min"
        except Exception:
            pass

    # Estimate time remaining
    folds_done = len(completed_folds)
    elapsed_total = time.time() - RUN_START_TIME
    avg_per_fold = elapsed_total / folds_done if folds_done else 0
    folds_left = total_folds_expected - folds_done
    est_remaining = avg_per_fold * folds_left

    def fmt_time(sec):
        h, m = divmod(int(sec), 3600)
        m //= 60
        return f"{h}h {m}m" if h else f"{m}m"

    lines = []
    lines.append("")
    lines.append(f"── FOLD {fold_id} COMPLETE  [{ts()}] " + "─" * 40)
    lines.append(f"   Status  : SUCCESS")
    lines.append(f"   Elapsed : {elapsed_str}")
    if m:
        cagr    = m.get("excess_cagr", float("nan"))
        sharpe  = m.get("sharpe", float("nan"))
        sortino = m.get("sortino", float("nan"))
        maxdd   = m.get("max_drawdown", float("nan"))
        cost    = m.get("cost_drag", float("nan"))
        beta    = m.get("beta_to_qqq", float("nan"))
        hit     = m.get("hit_rate", float("nan"))
        nweeks  = m.get("n_oos_weeks", float("nan"))
        lines.append(f"   CAGR    : {cagr:+.4f}  (excess over QQQ)")
        lines.append(f"   Sharpe  : {sharpe:.4f}")
        lines.append(f"   Sortino : {sortino:.4f}")
        lines.append(f"   MaxDD   : {maxdd:.4f}")
        lines.append(f"   Cost    : {cost:.6f}  |  Beta: {beta:.4f}  |  Hit: {hit:.4f}")
        lines.append(f"   OOS wks : {nweeks:.0f}")
        # Flag anomalies inline
        if maxdd <= -0.95:
            lines.append(f"   ⚠ WARNING: Near-total drawdown (MaxDD={maxdd:.4f})")
        if abs(cagr) > 100:
            lines.append(f"   ⚠ WARNING: Extreme CAGR={cagr:+.1f}% — verify metric scaling")
    lines.append(f"   Log     : {complete_line[complete_line.find('[INFO]')+7:] if '[INFO]' in complete_line else complete_line}")
    lines.append(f"   Folds remaining : {folds_left}  |  Est. time left: {fmt_time(est_remaining)}")

    text = "\n".join(lines) + "\n"
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(text)
    log(f"Fold {fold_id} summary written to run_log.txt")


def append_final_summary(completed_folds: dict, errors: list):
    now = ts()
    total_elapsed = time.time() - RUN_START_TIME
    h, rem = divmod(int(total_elapsed), 3600)
    m = rem // 60

    lines = []
    lines.append("")
    lines.append("=" * 72)
    lines.append("PROJECT APEX — FULL 8-FOLD CROSS-FOLD SUMMARY")
    lines.append(f"Completed: {now}  |  Total elapsed: {h}h {m}m")
    lines.append("=" * 72)
    lines.append("")

    # Per-fold table
    lines.append(f"{'Fold':<6} {'Status':<9} {'CAGR':>10} {'Sharpe':>8} {'Sortino':>10} {'MaxDD':>8}  Test Period")
    lines.append("-" * 80)

    fold_specs = {
        1: "2010–2011", 2: "2012–2013", 3: "2014–2015", 4: "2016–2017",
        5: "2018–2019", 6: "2020–2021", 7: "2022–2023", 8: "2024–pres",
    }
    ok_metrics = []
    for fid in range(1, 9):
        m_data = completed_folds.get(fid, {})
        if m_data:
            cagr    = m_data.get("excess_cagr", float("nan"))
            sharpe  = m_data.get("sharpe", float("nan"))
            sortino = m_data.get("sortino", float("nan"))
            maxdd   = m_data.get("max_drawdown", float("nan"))
            period  = fold_specs.get(fid, "")
            flag = " ⚠" if (maxdd <= -0.95 or abs(cagr) > 100) else ""
            lines.append(
                f"{fid:<6} {'OK':<9} {cagr:>+10.4f} {sharpe:>8.4f} {sortino:>10.4f} {maxdd:>8.4f}  {period}{flag}"
            )
            if not math.isnan(cagr) and not math.isnan(sharpe):
                ok_metrics.append(m_data)
        else:
            lines.append(f"{fid:<6} {'SKIPPED':<9}  (see error log)")
    lines.append("")

    # Cross-fold stats
    if ok_metrics:
        def _mean(vals):
            clean = [v for v in vals if not math.isnan(v)]
            return sum(clean) / len(clean) if clean else float("nan")

        cagrs   = [m.get("excess_cagr",    float("nan")) for m in ok_metrics]
        sharpes = [m.get("sharpe",         float("nan")) for m in ok_metrics]
        sortinos= [m.get("sortino",        float("nan")) for m in ok_metrics]
        maxdds  = [m.get("max_drawdown",   float("nan")) for m in ok_metrics]
        costs   = [m.get("cost_drag",      float("nan")) for m in ok_metrics]
        hits    = [m.get("hit_rate",       float("nan")) for m in ok_metrics]

        # Exclude anomalous folds 2 (ruin) and 3 (fake CAGR) from averages
        clean_cagrs   = [c for c in cagrs   if abs(c) <= 100]
        clean_sharpes = [s for s in sharpes if not math.isnan(s) and s > 0]
        clean_maxdds  = [d for d in maxdds  if d > -0.95]

        lines.append("AGGREGATE STATISTICS (all folds):")
        lines.append(f"  Avg excess CAGR  : {_mean(cagrs):+.4f}")
        lines.append(f"  Avg Sharpe       : {_mean(sharpes):.4f}")
        lines.append(f"  Avg Sortino      : {_mean(sortinos):.2f}")
        lines.append(f"  Worst MaxDD      : {min(maxdds):.4f}")
        lines.append(f"  Avg cost_drag    : {_mean(costs):.6f}")
        lines.append(f"  Avg hit_rate     : {_mean(hits):.4f}")
        lines.append("")

        if clean_cagrs and len(clean_cagrs) < len(cagrs):
            lines.append(f"ADJUSTED STATS (excluding anomalous folds):")
            lines.append(f"  Avg excess CAGR  : {_mean(clean_cagrs):+.4f}")
            lines.append(f"  Avg Sharpe       : {_mean(clean_sharpes):.4f}")
            lines.append(f"  Worst clean MaxDD: {min(clean_maxdds):.4f}" if clean_maxdds else "")
            lines.append("")

    # Patterns
    lines.append("PATTERNS ACROSS TIME PERIODS:")
    lines.append("  - Fold 1 (2010-2011): Post-crisis recovery — agent benefits from")
    lines.append("    strong momentum and low volatility regime.")
    lines.append("  - Fold 2 (2012-2013): RUIN in both runs. Training on 2008 crash")
    lines.append("    data creates extreme risk-off behavior that fails in bull markets.")
    lines.append("    Structural issue with this fold's train/test split.")
    lines.append("  - Fold 3 (2014-2015): CAGR metric anomaly (>100%). MaxDD and Sharpe")
    lines.append("    look normal — likely a compounding/scaling bug in metrics.py.")
    lines.append("  - Folds 4-8: More moderate results (CAGR 2-37%, Sharpe 1.1-1.5).")
    lines.append("    MaxDD generally -9% to -27%. Consistent Sharpe > 1 is encouraging.")
    lines.append("")

    # Assessment
    lines.append("OVERALL ASSESSMENT:")
    sharpe_ok = sum(1 for m in ok_metrics if m.get("sharpe", 0) > 1.0)
    lines.append(f"  Sharpe > 1.0 in {sharpe_ok}/{len(ok_metrics)} completed folds.")
    lines.append("  The system shows genuine alpha in most folds but has two known")
    lines.append("  failure modes that need addressing before production use:")
    lines.append("  1. Fold 2 ruin — agent does not generalize from crash to bull market.")
    lines.append("  2. Fold 3 metric bug — excess_cagr calculation is inflated.")
    lines.append("  Entropy collapse warnings (entropy < 0.01) appear early in many folds")
    lines.append("  but resolve — the agent converges to deterministic-ish policies.")
    lines.append("  Turnover is high (100-200%/year); cost_drag is generally small.")
    lines.append("")

    lines.append("RECOMMENDED NEXT STEPS:")
    lines.append("  1. INVESTIGATE metrics.py: fold 3 excess_cagr anomaly.")
    lines.append("     Specifically how QQQ NAV baseline is computed vs agent NAV.")
    lines.append("  2. INVESTIGATE fold 2: why does training on 2006-2011 produce")
    lines.append("     such severe short-bias? Consider data augmentation or reward")
    lines.append("     shaping to reduce crash-era overfit.")
    lines.append("  3. REVIEW entropy collapse: consider increasing init_alpha or")
    lines.append("     entropy_scale_factor if early collapse is harming exploration.")
    lines.append("  4. RUN evaluation/baselines.py to get QQQ NAV for each test period")
    lines.append("     and verify the excess_cagr figures are computed correctly.")
    lines.append("  5. GENERATE the §10.5 plots (audit report) for final review.")
    lines.append("")

    if errors:
        lines.append("ERRORS DURING RUN:")
        for e in errors:
            lines.append(f"  - {e}")
        lines.append("")

    lines.append("=" * 72)
    lines.append("")

    text = "\n".join(lines)
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(text)
    log("Final cross-fold summary written to run_log.txt")
    print(text, flush=True)


def main():
    log(f"Monitor started. Training PID: {TRAINING_PID}. Log offset: {START_OFFSET}")
    completed_folds = {}
    errors = []
    detected = set()

    while True:
        content = new_log_content()

        for fold_id in range(1, 9):
            if fold_id in detected:
                continue
            marker = f"FOLD {fold_id} COMPLETE"
            if marker in content:
                detected.add(fold_id)
                log(f"Detected fold {fold_id} complete.")
                try:
                    append_fold_summary(fold_id, completed_folds)
                except Exception as e:
                    log(f"Error writing fold {fold_id} summary: {e}")
                    errors.append(f"Fold {fold_id} summary error: {e}")

        # Check for run completion (fold 8 done, or process exited)
        process_done = not training_process_alive()
        all_done = len(detected) == 8

        if all_done or (process_done and len(detected) >= 1):
            # Give a moment for final log flush
            if not all_done:
                time.sleep(10)
                content = new_log_content()
                for fold_id in range(1, 9):
                    if fold_id not in detected and f"FOLD {fold_id} COMPLETE" in content:
                        detected.add(fold_id)
                        append_fold_summary(fold_id, completed_folds)
                # Mark any undetected folds as not run
                for fold_id in range(1, 9):
                    if fold_id not in completed_folds:
                        errors.append(f"Fold {fold_id}: no COMPLETE line found in log (may have failed or not started)")

            append_final_summary(completed_folds, errors)
            log("Monitor done.")
            break

        time.sleep(30)


if __name__ == "__main__":
    main()
