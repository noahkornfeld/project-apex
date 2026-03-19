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
\\- Confirm cuDNN benchmark is True and deterministic is False in run\\\_full\\\_apex.py    
\\- Confirm data files in Ticker\\\_Data/ are not empty stubs (each parquet \\\> 1KB)    
\\- python \\-m pytest tests/ \\-x \\-q

\\\#\\\# Known Issues Already Fixed (Do Not Re-Fix These)    
\\- Training speed (\\\~43s/update): FIXED — cuDNN benchmark mode    
\\- Unicode logging error on Windows: FIXED — special chars removed    
\\- Q-value divergence / fatal crash on 2008 data: FIXED — hyperparams tuned    
\\- Fold 3 crash at update 1250 (zero active assets): FIXED —    
  trading\\\_env.py force-liquidation logic updated

\\\#\\\# Known Issues Still to Watch For    
\\- F\\\_TOTAL mismatch (features/\\\_\\\_init\\\_\\\_.py may say 26, should be 25):    
  Fix by removing 'adj\\\_close' from TS\\\_FEATURE\\\_NAMES, set F\\\_TOTAL \\= 25    
\\- If you see any tensor dimension mismatch involving features,    
  check F\\\_TOTAL first — this is the most likely cause    
\\- vol\\\_52w min\\\_periods (per\\\_asset\\\_features.py \\\~line 122): if you see    
  noisy volatility warnings, change min\\\_periods=DAYS\\\_4W to    
  min\\\_periods=DAYS\\\_52W//2

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
\\- Key metrics: CAGR, Sharpe, Sortino, MaxDD    
\\- Any fixes applied and why    
\\- Estimated time remaining for the full run    
\\- What you recommend doing next

When all 8 folds are done, append a final cross-fold summary covering:    
\\- Overall results across all folds    
\\- Any patterns noticed (e.g. certain time periods harder than others)    
\\- What you recommend doing next

\\\#\\\# What NOT to Do    
\\- Do not revert cuDNN settings or any of the tuned hyperparameters    
\\- Do not install new Python packages without asking first    
\\- Do not delete or overwrite any data files in Ticker\\\_Data/    
\\- Do not modify master\\\_config.yaml values without explicit instruction    
