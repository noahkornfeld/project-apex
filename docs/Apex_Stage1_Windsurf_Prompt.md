# PROJECT APEX — STAGE 1 PROMPT (PASTE INTO WINDSURF)

## YOUR JOB

Write a single Python script called `run_full_apex.py` in the project root. This script will be run by the user overnight with `python run_full_apex.py` and must complete the full 8-fold training run entirely autonomously — no human input required at any point once it starts. Windsurf will not be involved after the script is running.

Do NOT run the training yourself. Do NOT execute code interactively. Just write the script.

---

## WHAT THE SCRIPT MUST DO

The script runs 8 sequential walk-forward folds. For each fold it:

1. Initializes a **fresh model** (no weight transfer between folds)
2. Runs **3 episodes** of training using the existing SAC components
3. Saves a **checkpoint** after the fold completes
4. Runs **OOS evaluation** on the fold's test window
5. Saves **per-fold results** to disk
6. Logs everything to a **persistent log file** so progress is visible in the morning

After all 8 folds, the script prints and saves a cross-fold summary table.

---

## EXISTING COMPONENTS TO USE

Wire together these already-written modules. Do not rewrite them:

```
config/config_loader.py          → load_config("config/master_config.yaml")
config/config_schema.py          → ProjectConfig dataclass
model/apex_actor_critic.py       → ApexActorCritic, from_config()
model/tcn.py                     → CausalTCN (used internally by actor-critic)
model/attention.py               → CrossAssetAttention (used internally)
environment/trading_env.py       → TradingEnvironment
environment/reward_fn.py         → RewardFunction, from_config()
training/replay_buffer.py        → ReplayBuffer, from_config()
utils/seed_utils.py              → set_global_seed(), get_episode_seed(), get_fold_seed()
scripts/build_all_panels.py      → run via subprocess if panels not yet built
features/feature_panel.py        → FeaturePanelBuilder, get_observation()
```

Load all hyperparameters from `config/master_config.yaml` via `load_config()`. Do not hardcode any values.

---

## FOLD DEFINITIONS

Load these from config. For reference:

| Fold | Train Start | Train End | Test Start | Test End |
|---|---|---|---|---|
| 1 | 2005-01-01 | 2009-12-31 | 2010-01-01 | 2011-12-31 |
| 2 | 2006-01-01 | 2011-12-31 | 2012-01-01 | 2013-12-31 |
| 3 | 2008-01-01 | 2013-12-31 | 2014-01-01 | 2015-12-31 |
| 4 | 2010-01-01 | 2015-12-31 | 2016-01-01 | 2017-12-31 |
| 5 | 2012-01-01 | 2017-12-31 | 2018-01-01 | 2019-12-31 |
| 6 | 2014-01-01 | 2019-12-31 | 2020-01-01 | 2021-12-31 |
| 7 | 2016-01-01 | 2021-12-31 | 2022-01-01 | 2023-12-31 |
| 8 | 2018-01-01 | 2023-12-31 | 2024-01-01 | present |

Embargo = 4 weeks between train_end and test_start. This is already set in config.

---

## TRAINING LOOP LOGIC (PER FOLD)

```python
for fold in range(1, 9):
    set seed: get_fold_seed(base_seed=42, fold_id=fold)
    initialize fresh model (from_config)
    initialize fresh optimizers (Adam/AdamW per config)
    initialize target networks (hard copy of model weights)
    load panel: Ticker_Data/panels/fold_{fold}/feature_panel.npz

    for episode in range(1, 4):
        set seed: get_episode_seed(42, fold, episode)
        reset ReplayBuffer
        reset RewardFunction
        
        # Warmup phase (52 steps, random feasible actions)
        for step in range(warmup_steps=52):
            sample random valid portfolio weights
            step environment
            store transition (marked is_warmup=True)
        
        # Training phase
        for each weekly step in training window:
            actor_forward() → w_pre
            project constraints → w_exec
            step environment → reward components
            compute reward via RewardFunction.compute()
            store transition in ReplayBuffer
            
            for update in range(updates_per_step=20):
                sample batch (size=64) from ReplayBuffer
                critic update (every update)
                actor + alpha update (every policy_delay=2 updates)
                polyak update target networks (tau=0.005)
            
            log every 250 updates
        
        replay_buffer.flush_episode()
    
    save checkpoint: checkpoints/fold_{fold}/model_final.pt
    run OOS evaluation on test window
    save results: results/fold_{fold}/
    log fold completion
```

---

## OPTIMIZER SETUP

```
Critic (Q1, Q2):  Adam,  lr=3e-4, weight_decay=0.0
Actor:            Adam,  lr=1e-4, weight_decay=0.0
Encoder (TCN+Attn): AdamW, lr=3e-4, weight_decay=1e-4
                    NOTE: weight_decay=0 for embedding params only
Alpha (log_alpha):  Adam, lr=1e-4
```

---

## HARDWARE CONSTRAINTS

- Single GPU only: RTX 3070, **8 GB VRAM**
- `mixed_precision = false` — full float32, no exceptions
- `distributed_training = false`
- If CUDA OOM occurs: catch the exception, reduce batch_size 64→32, retry the fold from episode 1, log the OOM event

---

## OUTPUT FILES THE SCRIPT MUST PRODUCE

```
checkpoints/
  fold_{N}/model_final.pt           ← saved after each fold completes

results/
  fold_{N}/
    oos_returns.csv                 ← weekly: date, portfolio_return, qqq_return
    oos_metrics.json                ← CAGR, Sharpe, Sortino, MaxDD, Turnover
    training_log.csv                ← per-update: update_num, q1_loss, q2_loss,
                                       actor_loss, alpha, entropy, td_error
  cross_fold/
    run_summary.txt                 ← printed table + pass/fail (written at end)

logs/
  apex_run.log                      ← single persistent log file, appended
                                       throughout the entire run
```

---

## LOGGING REQUIREMENTS

All log output goes to both console AND `logs/apex_run.log` simultaneously (use `logging` module with both StreamHandler and FileHandler).

Log these events:
- Script start: timestamp, GPU name, config loaded confirmation
- Each fold start: fold number, train/test date range
- Every 250 gradient updates: fold, episode, update count, q1_mean, q2_mean, entropy, alpha, td_error_mean
- Each episode end: episode reward total, turnover
- Each fold end: fold number, OOS CAGR, OOS Sharpe, checkpoint saved path
- Any alarm trigger (see below)
- Script end: total elapsed time, all-fold summary

---

## CRITICAL ALARMS — HALT AND LOG IF THESE OCCUR

| Condition | Action |
|---|---|
| Any w_exec > 0 where mask=0 (inactive slot) | CRITICAL — halt fold, log, skip to next fold |
| NaN anywhere in observations | CRITICAL — halt fold, log, skip to next fold |
| `\|\|w_exec\|\|₁` deviates from 1.0 by more than 1e-5, or any w_exec < 0 | CRITICAL — halt fold, log, skip to next fold |
| `\|q1_mean\|` or `\|q2_mean\|` > 100 | WARNING — log and continue |
| entropy_mean < 0.01 for 100+ consecutive steps | WARNING — log and continue |

---

## PRE-FLIGHT CHECKS (RUN BEFORE FOLD 1)

Before any training, the script must verify:
1. `torch.cuda.is_available()` is True — abort if not
2. `load_config("config/master_config.yaml")` validates without error — abort if not
3. All required parquet files exist in `Ticker_Data/` — abort with clear message if missing
4. Feature panels exist at `Ticker_Data/panels/fold_1/` through `fold_8/` — if missing, run `python scripts/build_all_panels.py` as a subprocess before proceeding

---

## FINAL SUMMARY TABLE

At the end of the run, print this to console and save to `results/cross_fold/run_summary.txt`:

```
╔══════════════════════════════════════════════════════════════╗
║            PROJECT APEX — 8-FOLD RUN COMPLETE               ║
╠═════╦══════════╦══════════╦═════════╦═════════╦═════════════╣
║Fold ║ CAGR (P) ║ CAGR (Q) ║ Sharpe  ║ Sortino ║  Max DD     ║
╠═════╬══════════╬══════════╬═════════╬═════════╬═════════════╣
║  1  ║  xx.x%   ║  xx.x%   ║  x.xx   ║  x.xx   ║  -xx.x%     ║
║  2  ║  ...     ║  ...     ║  ...    ║  ...    ║  ...        ║
║  8  ║  xx.x%   ║  xx.x%   ║  x.xx   ║  x.xx   ║  -xx.x%     ║
╠═════╩══════════╩══════════╩═════════╩═════════╩═════════════╣
║  Total elapsed: XX hours XX minutes                         ║
╚══════════════════════════════════════════════════════════════╝
```

---

## WHAT NOT TO DO

- Do not carry model weights from one fold to the next
- Do not use the HDD (secondary drive) for any file I/O — NVMe only
- Do not enable mixed_precision
- Do not modify any existing module files
- Do not hardcode hyperparameters — read everything from config
- Do not train on test-window data or use it for normalization
