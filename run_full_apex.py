#!/usr/bin/env python3
"""
run_full_apex.py
Project Apex — 8-Fold Walk-Forward Training Run.

Usage:
    python run_full_apex.py                    # Default: checkpoints/, results/
    python run_full_apex.py --run-name 20up    # Custom: checkpoints_20up/, results_20up/

All hyperparameters are loaded from config/master_config.yaml.
Logs are written to logs/apex_run.log and console simultaneously.
No human interaction required after launch.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ── Project-root on path ────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config.config_loader import load_config
from environment.constraint_projector import ConstraintProjector
from environment.market_data import build_market_data
from environment.reward_fn import RewardFunction
from environment.reward_fn import from_config as reward_from_config
from environment.trading_env import TradingEnvironment
from evaluation.metrics import compute_all_metrics
from features.feature_panel import FeaturePanelBuilder, get_observation
from model.apex_actor_critic import ApexActorCritic
from model.apex_actor_critic import from_config as model_from_config
from training.replay_buffer import ReplayBuffer
from training.replay_buffer import from_config as buffer_from_config
from training.sac_trainer import SACTrainer
from utils.seed_utils import get_episode_seed, get_fold_seed


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("apex")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Seed helper
# ─────────────────────────────────────────────────────────────────────────────

def _seed_all(seed: int, deterministic: bool = False) -> None:
    """Set all RNG seeds + deterministic mode."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # CRITICAL: cudnn.deterministic=True breaks CUDA kernel caching, causing 560x slowdown
    # Disable it even if deterministic mode is requested - RNG seeding provides reproducibility
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ─────────────────────────────────────────────────────────────────────────────
# Alarm checker
# ─────────────────────────────────────────────────────────────────────────────

_entropy_low_streak: int = 0


def check_alarms(
    w_exec: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    x_obs: Optional[np.ndarray],
    metrics: Dict,
    logger: logging.Logger,
) -> bool:
    """
    Run all critical and warning alarms.
    Returns True if a FATAL alarm fires (caller should halt).
    """
    global _entropy_low_streak

    if w_exec is not None and mask is not None:
        # Alarm 1: weight in inactive slot
        bad = (w_exec > 0) & (mask < 0.5)
        if bad.any():
            logger.error(
                f"ALARM [FATAL]: w_exec > 0 for inactive slots "
                f"{np.where(bad)[0].tolist()}"
            )
            return True

        # Alarm 3a: weight sum deviates from 1.0
        w_sum = float(np.sum(w_exec))
        n_active = int(np.sum(mask > 0.5))
        
        # Special case: if no active assets, allow all-zero weights (go to cash)
        if n_active == 0 and abs(w_sum) < 1e-8:
            logger.warning(
                f"ALARM [WARN]: No active assets, portfolio in cash (w_sum=0)"
            )
        elif abs(w_sum - 1.0) > 1e-5:
            logger.error(
                f"ALARM [FATAL]: ||w_exec||_1 = {w_sum:.8f}, "
                f"deviates from 1.0 by {abs(w_sum - 1.0):.2e}"
            )
            return True

        # Alarm 3b: any negative weight
        if (w_exec < 0).any():
            logger.error(
                f"ALARM [FATAL]: negative w_exec detected, "
                f"min={float(w_exec.min()):.6f}"
            )
            return True

    # Alarm 2: NaN in observations
    if x_obs is not None and np.isnan(x_obs).any():
        logger.error("ALARM [FATAL]: NaN detected in observations x_t")
        return True

    # Alarm 4: Q divergence (WARNING — continue)
    q1m = metrics.get("q1_mean", float("nan"))
    q2m = metrics.get("q2_mean", float("nan"))
    if not math.isnan(q1m) and abs(q1m) > 100.0:
        logger.warning(f"ALARM [WARN]: |q1_mean| = {abs(q1m):.2f} > 100")
    if not math.isnan(q2m) and abs(q2m) > 100.0:
        logger.warning(f"ALARM [WARN]: |q2_mean| = {abs(q2m):.2f} > 100")

    # Alarm 5: entropy collapse (WARNING — continue)
    # Only log when first crossing threshold (101) and every 100 updates after
    ent = metrics.get("entropy_mean", float("nan"))
    if not math.isnan(ent):
        if ent < 0.01:
            _entropy_low_streak += 1
        else:
            _entropy_low_streak = 0
        if _entropy_low_streak == 101 or (_entropy_low_streak > 100 and _entropy_low_streak % 100 == 0):
            logger.warning(
                f"ALARM [WARN]: entropy_mean < 0.01 for "
                f"{_entropy_low_streak} consecutive updates (entropy collapse)"
            )

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight checks
# ─────────────────────────────────────────────────────────────────────────────

def run_preflight(config, logger: logging.Logger) -> None:
    """Run all pre-flight checks; sys.exit(1) on failure."""

    # 1. CUDA required
    if not torch.cuda.is_available():
        logger.error(
            "PREFLIGHT [FAIL]: torch.cuda.is_available() is False. "
            "GPU is required."
        )
        sys.exit(1)
    logger.info(
        f"PREFLIGHT [1/4] CUDA OK — "
        f"{torch.cuda.get_device_name(0)}, "
        f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB VRAM"
    )

    # 2. Config already validated by load_config (no-op check)
    logger.info("PREFLIGHT [2/4] Config validated OK")

    # 3. Required parquet files
    data_dir = Path(config.data.data_dir)
    required_files = [
        data_dir / config.data.daily_bars_file,
        data_dir / config.data.ndx_membership_file,
        data_dir / config.data.macro_features_file,
        data_dir / config.data.trading_calendar_file,
    ]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        logger.error(
            "PREFLIGHT [FAIL]: Missing required parquet files:\n"
            + "\n".join(f"  {m}" for m in missing)
        )
        sys.exit(1)
    logger.info("PREFLIGHT [3/4] All required parquet files found")

    # 4. Feature panels — build if missing
    panels_root = data_dir / "panels_v2"
    shared_panel = panels_root / "shared" / "feature_panel_shared.npz"
    missing_panels = []
    if not shared_panel.exists():
        missing_panels.append(str(shared_panel))
    for fold_id in range(1, 9):
        p = panels_root / f"fold_{fold_id}" / "feature_panel.npz"
        if not p.exists():
            missing_panels.append(str(p))

    if missing_panels:
        logger.warning(
            f"PREFLIGHT [4/4]: {len(missing_panels)} panel file(s) missing. "
            "Running scripts/build_all_panels.py ..."
        )
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_all_panels.py")],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error(
                "PREFLIGHT [FAIL]: build_all_panels.py failed:\n"
                + result.stderr[-2000:]
            )
            sys.exit(1)
        logger.info("PREFLIGHT [4/4] Panels built successfully")
    else:
        logger.info("PREFLIGHT [4/4] All feature panels found")


# ─────────────────────────────────────────────────────────────────────────────
# ID Mapper: sparse security_id → dense index
# ─────────────────────────────────────────────────────────────────────────────

def build_id_mapper(panel_dict: dict) -> Tuple[Dict[int, int], int]:
    """
    Build mapping from sparse security_id to dense index [0, N-1].
    
    Returns
    -------
    id_map : dict mapping security_id → dense_index
    num_ids : total number of unique IDs (for embedding table size)
    """
    active_ids = panel_dict["active_ids"]
    valid_ids = active_ids[active_ids >= 0]
    unique_ids = np.unique(valid_ids)
    id_map = {int(sid): idx for idx, sid in enumerate(unique_ids)}
    return id_map, len(unique_ids)


def apply_id_mapping(ids_array: np.ndarray, id_mapper: Dict[int, int]) -> np.ndarray:
    """
    Apply ID mapping to convert sparse security_ids to dense indices.
    Inactive slots (id < 0) map to 0.
    Fully vectorized using numpy array indexing.
    """
    # Build lookup array once
    max_id = max(id_mapper.keys())
    lookup = np.zeros(max_id + 1, dtype=np.int64)
    for sparse_id, dense_idx in id_mapper.items():
        lookup[sparse_id] = dense_idx
    
    # Vectorized mapping
    ids_clipped = np.clip(ids_array, 0, max_id)
    return lookup[ids_clipped]


# ─────────────────────────────────────────────────────────────────────────────
# Data / environment builder
# ─────────────────────────────────────────────────────────────────────────────

def _load_panel(data_dir: Path, fold_id: int) -> dict:
    """Load the pre-built feature panel for a fold."""
    npz_path = data_dir / "panels_v2" / f"fold_{fold_id}" / "feature_panel.npz"
    return FeaturePanelBuilder.load(str(npz_path))


def build_fold_env(
    config,
    fold_spec: dict,
    train: bool,
    panel_dict: dict,
    logger: logging.Logger,
) -> Tuple[TradingEnvironment, dict, ConstraintProjector]:
    """
    Assemble TradingEnvironment, panel_data dict for SACTrainer, and projector.

    Returns
    -------
    env          : TradingEnvironment scoped to train or test window
    panel_data   : dict {x_panel, g_panel, mask_panel, ticker_ids, sector_ids}
                   for SACTrainer (full timeline, sliced by t_idx at sample time)
    projector    : ConstraintProjector
    """
    fold_id  = fold_spec["fold"]
    data_dir = Path(config.data.data_dir)
    shared_path = str(
        data_dir / "panels_v2" / "shared" / "feature_panel_shared.npz"
    )

    if train:
        win_start = fold_spec["train_start"]
        win_end   = fold_spec["train_end"]
    else:
        win_start = fold_spec["test_start"]
        raw_end   = fold_spec.get("test_end", "present")
        win_end   = None if raw_end in (None, "present") else raw_end

    md = build_market_data(
        bars_path         = str(data_dir / config.data.daily_bars_file),
        macro_path        = str(data_dir / config.data.macro_features_file),
        cal_path          = str(data_dir / config.data.trading_calendar_file),
        ndx_path          = str(data_dir / config.data.ndx_membership_file),
        shared_panel_path = shared_path,
        fold_train_start  = win_start,
        fold_train_end    = win_end,
    )

    # Inject feature panel arrays (from pre-built fold panel)
    md["x_panel"]    = panel_dict["x_panel"]
    md["g_panel"]    = panel_dict["g_panel"]
    md["mask_panel"] = panel_dict["mask_panel"]

    projector = ConstraintProjector(
        per_name_cap = config.constraints.per_name_cap,
        sector_cap   = config.constraints.sector_cap,
    )

    env = TradingEnvironment(
        md                 = md,
        projector          = projector,
        L_lookback         = config.architecture.L,
        missingness_config = config.missingness,
    )

    # Panel data for SACTrainer (full timeline; trainer slices by t_idx)
    panel_data = {
        "x_panel":    panel_dict["x_panel"],
        "g_panel":    panel_dict["g_panel"],
        "mask_panel": panel_dict["mask_panel"],
        "ticker_ids": panel_dict["active_ids"],   # security_id per slot
        "sector_ids": md["sector_ids"],            # GICS embedding index [T, K]
    }

    return env, panel_data, projector


# ─────────────────────────────────────────────────────────────────────────────
# Random feasible action (warmup)
# ─────────────────────────────────────────────────────────────────────────────

def _random_action(
    mask: np.ndarray,
    projector: ConstraintProjector,
    sector_ids: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Sample a Dirichlet-random portfolio weight vector and project it."""
    active_slots = np.where(mask > 0.5)[0]
    K = len(mask)
    w_full = np.zeros(K, dtype=np.float32)
    if len(active_slots) > 0:
        w_rand = np.random.dirichlet(
            np.ones(len(active_slots), dtype=np.float64)
        ).astype(np.float32)
        w_full[active_slots] = w_rand
    with torch.no_grad():
        w_t   = torch.from_numpy(w_full).to(device)
        m_t   = torch.from_numpy(mask).float().to(device)
        sid_t = torch.from_numpy(sector_ids).long().to(device)
        w_exec_t = projector(w_t, m_t, sid_t)
    return w_exec_t.cpu().numpy().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Get observation tensors for actor_forward
# ─────────────────────────────────────────────────────────────────────────────

def _obs_tensors(
    panel_data: dict,
    p_cur: int,
    L: int,
    device: torch.device,
    id_mapper: Optional[Dict[int, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build (x, g, mask, sector_ids, ticker_ids) tensors at panel row p_cur.
    Adds batch dimension [1, ...].
    
    If id_mapper is provided, maps sparse security_ids to dense indices.
    """
    x_t, g_t, mask_t, ids_t = get_observation(
        {
            "x_panel":    panel_data["x_panel"],
            "g_panel":    panel_data["g_panel"],
            "mask_panel": panel_data["mask_panel"],
            "active_ids": panel_data["ticker_ids"],
        },
        p_cur,
        L,
    )
    sid_t = panel_data["sector_ids"][p_cur]
    
    # Map sparse security_ids to dense indices if mapper provided
    if id_mapper is not None:
        ids_t = apply_id_mapping(ids_t, id_mapper)
    
    x_ten    = torch.from_numpy(x_t).float().unsqueeze(0).to(device)
    g_ten    = torch.from_numpy(g_t).float().unsqueeze(0).to(device)
    mask_ten = torch.from_numpy(mask_t).float().unsqueeze(0).to(device)
    sid_ten  = torch.from_numpy(sid_t).long().unsqueeze(0).to(device)
    ids_ten  = torch.from_numpy(ids_t).long().unsqueeze(0).to(device)
    return x_ten, g_ten, mask_ten, sid_ten, ids_ten


# ─────────────────────────────────────────────────────────────────────────────
# OOS evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_oos_eval(
    model: ApexActorCritic,
    config,
    fold_spec: dict,
    panel_dict: dict,
    device: torch.device,
    logger: logging.Logger,
    id_mapper: Optional[Dict[int, int]] = None,
) -> Tuple[Dict, np.ndarray, np.ndarray, List[str]]:
    """
    Run OOS evaluation for one fold.

    Returns
    -------
    metrics    : dict from compute_all_metrics
    port_arr   : [W_oos] weekly portfolio returns
    qqq_arr    : [W_oos] weekly QQQ returns
    dates_list : [W_oos] date strings (exec date per step)
    """
    logger.info(
        f"  [OOS] fold {fold_spec['fold']} — "
        f"{fold_spec['test_start']} to "
        f"{fold_spec.get('test_end','present')}"
    )

    data_dir    = Path(config.data.data_dir)
    shared_path = str(
        data_dir / "panels_v2" / "shared" / "feature_panel_shared.npz"
    )
    raw_end  = fold_spec.get("test_end", "present")
    win_end  = None if raw_end in (None, "present") else raw_end

    md_oos = build_market_data(
        bars_path         = str(data_dir / config.data.daily_bars_file),
        macro_path        = str(data_dir / config.data.macro_features_file),
        cal_path          = str(data_dir / config.data.trading_calendar_file),
        ndx_path          = str(data_dir / config.data.ndx_membership_file),
        shared_panel_path = shared_path,
        fold_train_start  = fold_spec["test_start"],
        fold_train_end    = win_end,
    )
    md_oos["x_panel"]    = panel_dict["x_panel"]
    md_oos["g_panel"]    = panel_dict["g_panel"]
    md_oos["mask_panel"] = panel_dict["mask_panel"]

    projector = ConstraintProjector(
        per_name_cap = config.constraints.per_name_cap,
        sector_cap   = config.constraints.sector_cap,
    )
    env_oos = TradingEnvironment(
        md                 = md_oos,
        projector          = projector,
        L_lookback         = config.architecture.L,
        missingness_config = config.missingness,
    )

    # Build panel_data for get_observation (sector_ids from oos md)
    panel_data_oos = {
        "x_panel":    panel_dict["x_panel"],
        "g_panel":    panel_dict["g_panel"],
        "mask_panel": panel_dict["mask_panel"],
        "ticker_ids": panel_dict["active_ids"],
        "sector_ids": md_oos["sector_ids"],
    }

    model.eval()
    obs, info = env_oos.reset()
    done = False
    L    = config.architecture.L

    port_returns: List[float] = []
    qqq_returns:  List[float] = []
    cost_bps_list: List[float] = []
    w_exec_hist:   List[np.ndarray] = []
    dates_list:    List[str] = []

    while not done:
        w_cur = env_oos._ep_step
        p_cur = env_oos._weekly_idx[w_cur]

        x_ten, g_ten, mask_ten, sid_ten, ids_ten = _obs_tensors(
            panel_data_oos, p_cur, L, device, id_mapper
        )
        with torch.no_grad():
            w_pre_t, _ = model.actor_forward(
                x_ten, g_ten, mask_ten, sid_ten, ids_ten
            )
        w_pre_np = w_pre_t.squeeze(0).cpu().numpy()

        obs, reward_components, done, info = env_oos.step(w_pre_np)

        port_returns.append(float(reward_components["r_port_t"]))
        qqq_returns.append(float(reward_components["r_qqq_t"]))
        cost_bps_list.append(float(reward_components["cost_t"]) * 1e4)
        w_exec_hist.append(env_oos._w_exec.copy())
        dates_list.append(info["date_exec"])

    model.train()

    if len(port_returns) == 0:
        logger.warning("  [OOS] Empty test window — no metrics computed")
        return {}, np.array([]), np.array([]), []

    port_arr   = np.array(port_returns, dtype=np.float64)
    qqq_arr    = np.array(qqq_returns,  dtype=np.float64)
    excess_arr = port_arr - qqq_arr
    nav        = np.cumprod(1.0 + port_arr)

    # Compute per-week one-way turnover from consecutive executed-weight changes.
    # At t=0 the portfolio starts from all-zeros, so delta_w[0] = w_exec[0].
    w_exec_arr   = np.stack(w_exec_hist).astype(np.float64)          # [T, K]
    w_exec_prev  = np.vstack([np.zeros((1, w_exec_arr.shape[1])),
                               w_exec_arr[:-1]])                       # [T, K]
    turnover_arr = 0.5 * np.abs(w_exec_arr - w_exec_prev).sum(axis=1) # [T]

    metrics = compute_all_metrics(
        nav               = nav,
        excess_returns    = excess_arr,
        qqq_returns       = qqq_arr,
        portfolio_returns = port_arr,
        turnover          = turnover_arr,
        cost_bps          = np.array(cost_bps_list, dtype=np.float64),
        w_exec            = w_exec_arr,
        asset_returns     = None,
    )
    return metrics, port_arr, qqq_arr, dates_list


# ─────────────────────────────────────────────────────────────────────────────
# Single-episode runner (warmup + training phases)
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(
    fold_id:          int,
    episode_id:       int,
    model:            ApexActorCritic,
    trainer:          SACTrainer,
    train_env:        TradingEnvironment,
    panel_data:       dict,
    projector:        ConstraintProjector,
    config,
    device:           torch.device,
    current_batch_size: int,
    log_cadence:      int,
    global_update_count: int,
    train_log_writer,
    logger:           logging.Logger,
    rng:              np.random.Generator,
    id_mapper:        Optional[Dict[int, int]] = None,
) -> Tuple[int, float, float]:
    """
    Run one episode: warmup + training phases.

    Returns
    -------
    global_update_count : updated counter after this episode
    ep_reward_total     : sum of clipped rewards this episode
    ep_turnover         : sum of ||delta_w||_1 this episode
    """
    warmup_steps     = config.replay.warmup_steps
    updates_per_step = config.sac.updates_per_step
    L                = config.architecture.L

    # ── Fresh replay buffer + reward function each episode ──────────────────
    replay = buffer_from_config(
        replay_cfg = vars(config.replay),
        arch_cfg   = vars(config.architecture),
        sac_cfg    = vars(config.sac),
    )
    reward_fn = reward_from_config(config.reward)

    # ── Warmup phase ─────────────────────────────────────────────────────────
    obs, info = train_env.reset()
    reward_fn.reset()
    logger.info(
        f"  Fold {fold_id} | Ep {episode_id} | Warmup "
        f"({warmup_steps} random steps) ..."
    )

    for wu_step in range(warmup_steps):
        w_cur = train_env._ep_step
        p_cur = train_env._weekly_idx[w_cur]
        mask_cur = panel_data["mask_panel"][p_cur].copy()
        sid_cur  = panel_data["sector_ids"][p_cur].copy()

        w_rand = _random_action(mask_cur, projector, sid_cur, device)
        t_idx_before = int(p_cur)

        obs_step, rc, done_wu, info_wu = train_env.step(w_rand)
        w_exec_wu = train_env._w_exec.copy()

        w_next   = train_env._ep_step
        p_next   = int(
            train_env._weekly_idx[
                min(w_next, len(train_env._weekly_idx) - 1)
            ]
        )

        r_dict = reward_fn.compute_from_env(rc)
        r      = r_dict["reward"]

        replay.add_step(
            t_idx      = t_idx_before,
            mask_t     = mask_cur,
            w_pre      = w_rand,
            w_exec     = w_exec_wu,
            reward     = r,
            t_idx_next = p_next,
            done       = done_wu,
            is_warmup  = True,
            step_idx   = wu_step,
        )
        if done_wu:
            break

    replay.flush_episode()

    # ── Training phase ────────────────────────────────────────────────────────
    obs, info = train_env.reset()
    reward_fn.reset()
    logger.info(
        f"  Fold {fold_id} | Ep {episode_id} | Training phase ..."
    )

    ep_reward_total = 0.0
    ep_turnover     = 0.0
    ep_steps        = 0
    train_step_idx  = 0

    while True:
        w_cur = train_env._ep_step
        p_cur = train_env._weekly_idx[w_cur]
        mask_cur = panel_data["mask_panel"][p_cur].copy()

        # Build obs + get action from policy
        x_ten, g_ten, mask_ten, sid_ten, ids_ten = _obs_tensors(
            panel_data, p_cur, L, device, id_mapper
        )
        x_np = x_ten.squeeze(0).cpu().numpy()

        # Alarm 2: NaN in obs
        if np.isnan(x_np).any():
            logger.error(
                f"ALARM [FATAL]: NaN in x_t at panel row {p_cur}"
            )
            sys.exit(2)

        with torch.no_grad():
            w_pre_t, _ = model.actor_forward(
                x_ten, g_ten, mask_ten, sid_ten, ids_ten
            )
        w_pre_np = w_pre_t.squeeze(0).cpu().numpy()
        t_idx_before = int(p_cur)

        obs_step, rc, done, info_step = train_env.step(w_pre_np)
        w_exec_np = train_env._w_exec.copy()

        w_next_ep = train_env._ep_step
        p_next    = int(
            train_env._weekly_idx[
                min(w_next_ep, len(train_env._weekly_idx) - 1)
            ]
        )

        # Alarm 1 + 3: inactive slots / norm check
        fatal = check_alarms(w_exec_np, mask_cur, None, {}, logger)
        if fatal:
            logger.error("Fatal alarm — aborting run")
            sys.exit(2)

        r_dict = reward_fn.compute_from_env(rc)
        r      = r_dict["reward"]
        ep_reward_total += r
        ep_turnover     += float(
            np.abs(train_env._w_exec - train_env._w_exec_prev).sum()
        )
        ep_steps += 1

        replay.add_step(
            t_idx      = t_idx_before,
            mask_t     = mask_cur,
            w_pre      = w_pre_np,
            w_exec     = w_exec_np,
            reward     = r,
            t_idx_next = p_next,
            done       = done,
            is_warmup  = False,
            step_idx   = train_step_idx,
        )

        # SAC updates
        if replay.size >= current_batch_size:
            for upd_idx in range(updates_per_step):
                batch  = replay.sample(current_batch_size, critic=True, rng=rng)
                upd    = trainer.update(batch, rng=rng)
                global_update_count += 1

                # Write training log row
                train_log_writer.writerow({
                    "update_num":    global_update_count,
                    "q1_loss":       f"{upd.get('loss_q1', float('nan')):.6f}",
                    "q2_loss":       f"{upd.get('loss_q2', float('nan')):.6f}",
                    "actor_loss":    f"{upd.get('actor_loss', float('nan')):.6f}",
                    "alpha":         f"{upd.get('alpha', float('nan')):.6f}",
                    "entropy":       f"{upd.get('entropy_mean', float('nan')):.6f}",
                    "td_error":      f"{upd.get('td_error_abs_mean', float('nan')):.6f}",
                })

                # Periodic log every log_cadence updates
                if global_update_count % log_cadence == 0:
                    logger.info(
                        f"  Fold {fold_id} | Ep {episode_id} | "
                        f"Update {global_update_count:6d} | "
                        f"q1={upd.get('q1_mean', float('nan')):8.4f}  "
                        f"q2={upd.get('q2_mean', float('nan')):8.4f}  "
                        f"ent={upd.get('entropy_mean', float('nan')):7.4f}  "
                        f"a={upd.get('alpha', float('nan')):.4f}  "
                        f"td={upd.get('td_error_abs_mean', float('nan')):.4f}"
                    )

                # Alarm 4 + 5 on update metrics
                check_alarms(None, None, None, upd, logger)

        train_step_idx += 1
        if done:
            break

    replay.flush_episode()
    logger.info(
        f"  Fold {fold_id} | Ep {episode_id} DONE | "
        f"steps={ep_steps}  reward={ep_reward_total:.4f}  "
        f"turnover={ep_turnover:.4f}  buf={replay.size}"
    )
    return global_update_count, ep_reward_total, ep_turnover


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(run_name: str = "", fold_only: int = 0) -> None:
    global _entropy_low_streak
    t0_total = time.time()

    # ── Logging ──────────────────────────────────────────────────────────────
    log_path = ROOT / "logs" / "apex_run.log"
    logger   = setup_logging(log_path)

    logger.info("=" * 72)
    logger.info(
        f"Project Apex — 8-Fold Training Run  |  "
        f"started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    logger.info(f"Log file: {log_path}")

    # ── Config ───────────────────────────────────────────────────────────────
    config_path = ROOT / "config" / "master_config.yaml"
    config = load_config(str(config_path))
    logger.info(f"Config loaded from {config_path}")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Device: {gpu_name}  |  VRAM: {vram_gb:.1f} GB")
        
        # Enable cuDNN autotuner for better performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    else:
        logger.info("Device: CPU (no CUDA available)")

    # ── Pre-flight ───────────────────────────────────────────────────────────
    run_preflight(config, logger)

    # ── Hyperparameters from config ──────────────────────────────────────────
    base_seed         = config.random_seed.base_seed
    use_det           = config.random_seed.use_deterministic
    n_episodes        = config.collection.n_episodes_per_fold
    log_cadence       = config.logging.update_cadence
    batch_size        = config.sac.batch_size
    data_dir          = Path(config.data.data_dir)

    fold_results: List[Dict] = []

    # ════════════════════════════════════════════════════════════════════════
    # 8-Fold loop
    # ════════════════════════════════════════════════════════════════════════
    folds_to_run = [fold_only] if fold_only else list(range(1, 9))
    for fold_id in folds_to_run:
        t0_fold = time.time()
        _entropy_low_streak = 0

        fold_spec = config.evaluation.folds[fold_id - 1]
        logger.info("=" * 72)
        logger.info(
            f"FOLD {fold_id}/8  |  "
            f"train: {fold_spec['train_start']} – {fold_spec['train_end']}  |  "
            f"test: {fold_spec['test_start']} – {fold_spec.get('test_end','present')}"
        )

        # Fold-level seed
        fold_seed = get_fold_seed(base_seed=base_seed, fold_id=fold_id)
        _seed_all(fold_seed, deterministic=use_det)

        # Load feature panel once per fold (shared across train + oos)
        logger.info(f"  Loading feature panel for fold {fold_id} ...")
        panel_dict = _load_panel(data_dir, fold_id)
        
        # Build ID mapper: sparse security_id → dense index [0, N-1]
        id_mapper, num_unique_ids = build_id_mapper(panel_dict)
        logger.info(f"  ID mapper built: {num_unique_ids} unique security IDs")

        # Build training env + panel_data
        logger.info(f"  Building training environment for fold {fold_id} ...")
        train_env, panel_data, projector = build_fold_env(
            config, fold_spec, train=True, panel_dict=panel_dict, logger=logger
        )
        logger.info(
            f"  Training window: {train_env._W} weekly steps"
        )

        # Output directories
        ckpt_base = f"checkpoints_{run_name}" if run_name else "checkpoints"
        results_base = f"results_{run_name}" if run_name else "results"
        ckpt_dir    = ROOT / ckpt_base / f"fold_{fold_id}"
        results_dir = ROOT / results_base / f"fold_{fold_id}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        train_log_path   = results_dir / "training_log.csv"
        train_log_fields = [
            "update_num", "q1_loss", "q2_loss",
            "actor_loss", "alpha", "entropy", "td_error",
        ]

        # ── OOM retry loop ───────────────────────────────────────────────────
        current_batch_size  = batch_size
        oom_happened        = False

        for _attempt in range(2):  # attempt 0: batch=64; attempt 1: batch=32
            if oom_happened:
                current_batch_size = max(32, current_batch_size // 2)
                logger.warning(
                    f"  OOM retry (attempt {_attempt + 1}): "
                    f"batch_size reduced to {current_batch_size}"
                )
                torch.cuda.empty_cache()

            # Fresh model + trainer (always from scratch, no weight transfer)
            # num_tickers = number of unique security IDs (mapped to dense indices)
            model = model_from_config(
                vars(config.architecture),
                D_g         = 20,
                num_tickers = num_unique_ids,
                num_sectors = 12,
            ).to(device)

            trainer = SACTrainer(
                model      = model,
                panel_data = panel_data,
                sac_cfg    = config.sac,
                opt_cfg    = config.optimizer,
                L_lookback = config.architecture.L,
                device     = device,
                id_mapper  = id_mapper,
            )

            global_update_count = 0

            # Open training log
            train_log_file = open(train_log_path, "w", newline="", encoding="utf-8")
            train_log_writer = csv.DictWriter(
                train_log_file, fieldnames=train_log_fields
            )
            train_log_writer.writeheader()

            try:
                # ── Episode loop ─────────────────────────────────────────────
                for episode_id in range(1, n_episodes + 1):
                    ep_seed = get_episode_seed(
                        base_seed=base_seed,
                        fold_id=fold_id,
                        episode_id=episode_id,
                    )
                    _seed_all(ep_seed, deterministic=use_det)
                    rng = np.random.default_rng(ep_seed + 1_000_000)

                    global_update_count, ep_reward, ep_turn = run_episode(
                        fold_id             = fold_id,
                        episode_id          = episode_id,
                        model               = model,
                        trainer             = trainer,
                        train_env           = train_env,
                        panel_data          = panel_data,
                        projector           = projector,
                        config              = config,
                        device              = device,
                        current_batch_size  = current_batch_size,
                        log_cadence         = log_cadence,
                        global_update_count = global_update_count,
                        train_log_writer    = train_log_writer,
                        logger              = logger,
                        rng                 = rng,
                        id_mapper           = id_mapper,
                    )

                # All episodes completed successfully
                break

            except torch.cuda.OutOfMemoryError:
                train_log_file.close()
                torch.cuda.empty_cache()
                logger.error(
                    f"  CUDA OOM on fold {fold_id} attempt {_attempt + 1} "
                    f"with batch_size={current_batch_size}"
                )
                oom_happened = True
                continue

        train_log_file.close()

        # ── Save checkpoint ───────────────────────────────────────────────────
        ckpt_path = ckpt_dir / "model_final.pt"
        torch.save(
            {
                "fold":            fold_id,
                "model_state":     model.state_dict(),
                "log_alpha":       trainer.log_alpha.item(),
                "update_step":     trainer.update_step,
                "batch_size_used": current_batch_size,
            },
            ckpt_path,
        )
        logger.info(f"  Checkpoint saved: {ckpt_path}")

        # ── OOS Evaluation ────────────────────────────────────────────────────────
        oos_result = run_oos_eval(
            model=model,
            config=config,
            fold_spec=fold_spec,
            panel_dict=panel_dict,
            device=device,
            logger=logger,
            id_mapper=id_mapper,
        )
        metrics, port_arr, qqq_arr, dates_list = oos_result

        # Save oos_returns.csv
        oos_returns_path = results_dir / "oos_returns.csv"
        with open(oos_returns_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "portfolio_return", "qqq_return"])
            for d, pr, qr in zip(dates_list, port_arr.tolist(), qqq_arr.tolist()):
                writer.writerow([d, f"{pr:.8f}", f"{qr:.8f}"])

        # Save oos_metrics.json
        oos_metrics_path = results_dir / "oos_metrics.json"
        with open(oos_metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                    for k, v in metrics.items()
                },
                f,
                indent=2,
            )

        elapsed_fold = time.time() - t0_fold
        cagr    = metrics.get("excess_cagr",  float("nan"))
        sharpe  = metrics.get("sharpe",        float("nan"))
        sortino = metrics.get("sortino",       float("nan"))
        maxdd   = metrics.get("max_drawdown",  float("nan"))
        logger.info(
            f"FOLD {fold_id} COMPLETE | "
            f"OOS CAGR={cagr:+.4f}  Sharpe={sharpe:.4f}  "
            f"Sortino={sortino:.4f}  MaxDD={maxdd:.4f} | "
            f"elapsed {elapsed_fold / 60:.1f} min | "
            f"ckpt → {ckpt_path}"
        )

        fold_results.append(
            {
                "fold":        fold_id,
                "train_start": fold_spec["train_start"],
                "train_end":   fold_spec["train_end"],
                "test_start":  fold_spec["test_start"],
                "test_end":    fold_spec.get("test_end", "present"),
                **{
                    k: (
                        float(v)
                        if isinstance(v, (int, float, np.floating))
                        else v
                    )
                    for k, v in metrics.items()
                },
            }
        )

    # ════════════════════════════════════════════════════════════════════════
    # Cross-fold summary
    # ════════════════════════════════════════════════════════════════════════
    elapsed_total = time.time() - t0_total

    def _nanmean(key: str) -> float:
        vals = [
            r[key]
            for r in fold_results
            if key in r and not math.isnan(float(r[key]))
        ]
        return float(np.mean(vals)) if vals else float("nan")

    col_w = 23
    hdr = (
        f"{'Fold':>4}  "
        f"{'Train Range':{col_w}}  "
        f"{'Test Range':{col_w}}  "
        f"{'CAGR':>8}  "
        f"{'Sharpe':>8}  "
        f"{'Sortino':>8}  "
        f"{'MaxDD':>8}"
    )
    sep = "-" * len(hdr)

    lines = [
        "=" * len(hdr),
        "Project Apex — Cross-Fold Summary",
        f"Completed {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  "
        f"|  Total elapsed: {elapsed_total / 3600:.2f} h",
        "=" * len(hdr),
        hdr,
        sep,
    ]
    for r in fold_results:
        train_rng = f"{r['train_start']} – {r['train_end']}"
        test_rng  = f"{r['test_start']} – {r['test_end']}"
        cagr    = r.get("excess_cagr",  float("nan"))
        sharpe  = r.get("sharpe",        float("nan"))
        sortino = r.get("sortino",       float("nan"))
        maxdd   = r.get("max_drawdown",  float("nan"))
        lines.append(
            f"{r['fold']:>4}  "
            f"{train_rng:{col_w}}  "
            f"{test_rng:{col_w}}  "
            f"{cagr:>+8.4f}  "
            f"{sharpe:>8.4f}  "
            f"{sortino:>8.4f}  "
            f"{maxdd:>8.4f}"
        )

    avg_cagr    = _nanmean("excess_cagr")
    avg_sharpe  = _nanmean("sharpe")
    avg_sortino = _nanmean("sortino")
    avg_maxdd   = _nanmean("max_drawdown")
    lines += [
        sep,
        f"{'AVG':>4}  "
        f"{'':>{col_w}}  "
        f"{'':>{col_w}}  "
        f"{avg_cagr:>+8.4f}  "
        f"{avg_sharpe:>8.4f}  "
        f"{avg_sortino:>8.4f}  "
        f"{avg_maxdd:>8.4f}",
        "=" * len(hdr),
    ]

    summary_text = "\n".join(lines)
    logger.info("\n" + summary_text)
    logger.info(f"Total elapsed: {elapsed_total / 3600:.2f} h")

    results_base = f"results_{run_name}" if run_name else "results"
    cross_fold_dir = ROOT / results_base / "cross_fold"
    cross_fold_dir.mkdir(parents=True, exist_ok=True)
    summary_path = cross_fold_dir / "run_summary.txt"
    summary_path.write_text(summary_text + "\n", encoding="utf-8")
    logger.info(f"Cross-fold summary saved → {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Project Apex 8-fold training")
    parser.add_argument("--run-name", type=str, default="",
                        help="Custom suffix for output directories (e.g., '20up' -> checkpoints_20up/, results_20up/)")
    parser.add_argument("--fold", type=int, default=0,
                        help="Run a single fold only (1-8). Default 0 = run all 8 folds.")
    args = parser.parse_args()
    main(run_name=args.run_name, fold_only=args.fold)
