\\\# Project Apex \\- Claude Code Instructions

\\\#\\\# What This Project Is    
This is Project Apex: a reinforcement learning trading agent that manages    
a Nasdaq-100 portfolio. It uses SAC (Soft Actor-Critic) training across    
8 time-period folds, benchmarked against QQQ. The master specification    
is Bible v5. Do not change any hyperparameters without being told to.

\\\#\\\# How to Start a Run    
Full 8-fold run (default):    
python run\\\_full\\\_apex.py

Single fold only:    
python run\\\_full\\\_apex.py \\--fold 3

This is the correct entry point. Do NOT use e2e\\\_runner.py.

\\\#\\\# Your Job When I Start You    
1\\. Run the pre-run checklist (see below) and fix any issues before starting    
2\\. Confirm updates\\\_per\\\_step is set to 20 in config/master\\\_config.yaml    
3\\. Start the run I specify    
4\\. Monitor for errors and fix them autonomously if possible    
5\\. After each fold, append a summary to run\\\_log.txt    
6\\. If you cannot fix an error after 2 attempts, skip that fold,    
   log what happened clearly, and continue with the next fold —    
   do not stop the entire run because of one fold failure    
7\\. Never modify master\\\_config.yaml hyperparameters without asking me first

\\\#\\\# Pre-Run Checklist (Run Before Every Session)
\\- python \\-c "from config.config\\\_loader import load\\\_config; load\\\_config('config/master\\\_config.yaml'); print('Config OK')"
\\- python \\-c "import torch; print('GPU:', torch.cuda.is\\\_available())"
\\- Confirm updates\\\_per\\\_step is 20 in config/master\\\_config.yaml
\\- Confirm lambda\\\_slow is 0.50 in config/master\\\_config.yaml (reduced from 0.75, 2026-03-19)
\\- Confirm bellman\\\_clip\\\_low=-30.0 and bellman\\\_clip\\\_high=30.0 in config/master\\\_config.yaml
\\- Confirm cuDNN benchmark is True and deterministic is False in run\\\_full\\\_apex.py
\\- Confirm data files in Ticker\\\_Data/ are not empty stubs (each parquet \\\> 1KB)
\\- Confirm new feature names are active:
  python \\-c "from features.macro\\\_broadcast\\\_features import MACRO\\\_FEATURE\\\_NAMES; from features.benchmark\\\_features import BENCHMARK\\\_FEATURE\\\_NAMES; assert 'vix\\\_4w\\\_trend' in MACRO\\\_FEATURE\\\_NAMES; assert 'qqq\\\_ret\\\_52w' in BENCHMARK\\\_FEATURE\\\_NAMES; print('Features OK')"
\\- Confirm all 8 g\\\_panels have shape (5577, 20):
  python \\-c "import numpy as np; [print(f'Fold \\{i\\}: \\{np.load(f\"Ticker\\_Data/panels\\_v2/fold\\_\\{i\\}/feature\\_panel.npz\")[\"g\\_panel\"].shape\\}') for i in range(1,9)]"
\\- python \\-m pytest tests/ \\-x \\-q

\\\#\\\# Known Issues Already Fixed (Do Not Re-Fix These)
\\- Training speed (\\\~43s/update): FIXED — cuDNN benchmark mode
\\- Unicode logging error on Windows: FIXED — special chars removed
\\- Fold 3 crash at update 1250 (zero active assets): FIXED —
  trading\\\_env.py force-liquidation logic updated
\\- Extreme single-step return spikes in data: FIXED — ret\\\_asset clipped,
  robust beta, compounded cost\\\_drag (commit fa57f1e)
\\- Metrics bug (Sharpe/Sortino computed on excess returns = Information Ratio):
  FIXED in commit 5598e86 — both now computed on raw portfolio returns.
  Expected Sharpe range is 0.7-1.5, not 0.3 or 100+.
\\- Hurricane Sandy (Oct 29, 2012) zero-price row: FIXED —
  trading\\\_calendar.parquet sets is\\\_week\\\_start=False for that date;
  market\\\_data.py forward-fills any market-wide zero-price row.
  "No active assets" warnings in Folds 3-5 training should no longer appear.
\\- beta = -101 in Fold 5 (Run 2): was a data corruption artifact from
  corrupted adj\\\_close prices, not a model behavior. Fixed by fa57f1e.
  Current runs show beta 0.8-1.1 across all folds (long-only portfolio).
\\- vix\\\_change (5-day delta, no usable signal): REPLACED by vix\\\_4w\\\_trend
  (20-day delta) in features/macro\\\_broadcast\\\_features.py (2026-03-19).
  Do not revert to diff(5).
\\- qqq\\\_vol\\\_4w (4-week volatility): REPLACED by qqq\\\_ret\\\_52w (52-week
  return) in features/benchmark\\\_features.py (2026-03-19). Do not revert.
\\- All 8 g\\\_panels rebuilt with new features (2026-03-19). Do not rebuild
  from scratch unless per-asset x\\\_panel features also change.

\\\#\\\# Known Issues Still to Watch For
\\- F\\\_TOTAL mismatch (features/\\\_\\\_init\\\_\\\_.py may say 26, should be 25):
  Fix by removing 'adj\\\_close' from TS\\\_FEATURE\\\_NAMES, set F\\\_TOTAL \\= 25
\\- If you see any tensor dimension mismatch involving features,
  check F\\\_TOTAL first — this is the most likely cause
\\- vol\\\_52w min\\\_periods (per\\\_asset\\\_features.py \\\~line 122): if you see
  noisy volatility warnings, change min\\\_periods=DAYS\\\_4W to
  min\\\_periods=DAYS\\\_52W//2
\\- Fold 2 Ep3 Q-divergence (structural, partially mitigated): training on
  2006-2011 window causes bootstrapping instability at Ep3 ~update 14,250.
  Q-values reach 1000-1400, TD error spikes to 500+. Root cause: crash
  (2008-09) and recovery (2010-11) coexist in replay buffer; high-Q
  bootstrap targets from recovery period contaminate crash-period transitions.
  The data-pipeline fix (fa57f1e) is separate and still valid. The new
  Bellman target clip (bellman\\\_clip\\\_low=-30, bellman\\\_clip\\\_high=30 in
  master\\\_config.yaml) arrests the runaway loop. If divergence persists,
  reduce bellman\\\_clip\\\_high toward 20 — do NOT touch grad\\\_clip\\\_critic.

\\\#\\\# Error Handling Rules    
\\- If a fold fails: read the error, identify the cause, fix it, retry ONCE    
\\- If it fails a second time: skip that fold, log clearly what happened,    
  and continue with the next fold    
\\- If a fix requires changing core model architecture or Bible v5    
  hyperparameters, do NOT make that change — stop and ask me    
\\- Always APPEND to run\\\_log.txt — never overwrite (preserve history)

\\\#\\\# What to Report After Each Fold
Append to run\\\_log.txt after every fold (success or failure):
\\- Fold number and date/time completed
\\- Did it succeed or fail?
\\- How long did it take?
\\- Key metrics: excess CAGR, Sharpe (raw returns), Sortino (raw returns), MaxDD
  NOTE: Sharpe and Sortino must use RAW portfolio returns, not excess returns.
  The Information Ratio (Sharpe of excess returns) is a separate metric.
  Raw returns are in results/fold\\\_N/oos\\\_returns.csv column 'portfolio\\\_return'.
\\- Beta to QQQ (expected range 0.7-1.2 for a sound long-only run)
\\- Whether the Bellman clip fired (Q-values reaching \\+/-30 in training logs)
\\- Whether any "No active assets" warnings appeared, and on what date
\\- Any fixes applied and why
\\- Estimated time remaining for the full run
\\- What you recommend doing next

When all 8 folds are done, append a final cross-fold summary covering:
\\- Per-fold: raw Sharpe vs QQQ Sharpe, raw Sortino vs QQQ Sortino, excess CAGR
\\- Whether Folds 4, 5, 8 (previous bull-market underperformers) improved
\\- Any patterns noticed (e.g. certain time periods harder than others)
\\- What you recommend doing next

\\\#\\\# What NOT to Do
\\- Do not revert cuDNN settings or any of the tuned hyperparameters
\\- Do not revert lambda\\\_slow back to 0.75 — it is intentionally 0.50
\\- Do not revert vix\\\_4w\\\_trend back to vix\\\_change (diff(5)) — the 5-day
  delta was confirmed to carry no usable signal (p=0.80 across all folds)
\\- Do not revert qqq\\\_ret\\\_52w back to qqq\\\_vol\\\_4w
\\- Do not remove or change the Bellman target clip in sac\\\_trainer.py
\\- Do not reinstall Python packages without asking first
\\- Do not delete or overwrite any data files in Ticker\\\_Data/
\\- Do not rebuild the full panels from scratch unless x\\\_panel features change —
  only g\\\_panels need rebuilding when macro/benchmark features change
\\- Do not modify master\\\_config.yaml values without explicit instruction
\\- Do not report the Information Ratio as "Sharpe" — they are different metrics
