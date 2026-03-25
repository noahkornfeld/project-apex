"""
recompute_turnover.py
=====================
Re-runs OOS evaluation for all 8 folds using saved checkpoints and updates
oos_metrics.json in-place with an accurate turnover_mean value.

No training is performed. Each fold takes ~1-2 minutes (inference only).

Usage:
    python recompute_turnover.py                      # all 8 folds, run_3_20_26
    python recompute_turnover.py --folds 2 5 8        # specific folds only
    python recompute_turnover.py --run-name 3_20_26   # explicit results dir suffix
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.config_loader import load_config
from environment.market_data import build_market_data
from environment.constraint_projector import ConstraintProjector
from environment.trading_env import TradingEnvironment
from evaluation.metrics import compute_all_metrics
from features.feature_panel import FeaturePanelBuilder
from model.apex_actor_critic import from_config as model_from_config

from run_full_apex import (
    _obs_tensors,
    build_id_mapper,
    _load_panel,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OOS replay with turnover
# ---------------------------------------------------------------------------

def _oos_with_turnover(
    model,
    config,
    fold_spec: dict,
    panel_dict: dict,
    device: torch.device,
    id_mapper: Dict[int, int],
) -> dict:
    """Re-run OOS inference for one fold; return full metrics dict."""
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

    port_returns:  List[float]       = []
    qqq_returns:   List[float]       = []
    cost_bps_list: List[float]       = []
    w_exec_hist:   List[np.ndarray]  = []

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

    if not port_returns:
        logger.warning("  Empty OOS window — skipping")
        return {}

    port_arr   = np.array(port_returns,  dtype=np.float64)
    qqq_arr    = np.array(qqq_returns,   dtype=np.float64)
    excess_arr = port_arr - qqq_arr
    nav        = np.cumprod(1.0 + port_arr)

    w_stack      = np.stack(w_exec_hist).astype(np.float64)
    turnover_arr = 0.5 * np.sum(np.abs(np.diff(w_stack, axis=0)), axis=1)

    return compute_all_metrics(
        nav               = nav,
        excess_returns    = excess_arr,
        qqq_returns       = qqq_arr,
        portfolio_returns = port_arr,
        turnover          = turnover_arr,
        cost_bps          = np.array(cost_bps_list, dtype=np.float64),
        w_exec            = w_stack,
        asset_returns     = None,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute turnover for saved checkpoints")
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(1, 9)),
                        help="Fold IDs to process (default: 1-8)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Results dir suffix, e.g. '3_20_26' → results/run_3_20_26")
    parser.add_argument("--config", type=str, default="config/master_config.yaml",
                        help="Path to master config")
    parser.add_argument("--ckpt-base", type=str, default="checkpoints",
                        help="Checkpoint base directory (default: checkpoints)")
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = load_config(str(config_path))
    logger.info(f"Config loaded from {config_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    data_dir = Path(config.data.data_dir)

    # Resolve results directory
    if args.run_name:
        results_base = ROOT / f"results/run_{args.run_name}"
    else:
        # Default: find the most recently modified run directory
        run_dirs = sorted(
            (ROOT / "results").glob("run_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not run_dirs:
            logger.error("No run_* directories found under results/")
            sys.exit(1)
        results_base = run_dirs[0]
        logger.info(f"Auto-selected results dir: {results_base.name}")

    ckpt_base = ROOT / args.ckpt_base

    logger.info(f"Results dir : {results_base}")
    logger.info(f"Checkpoint dir: {ckpt_base}")
    logger.info(f"Folds to process: {args.folds}")

    for fold_id in args.folds:
        fold_spec    = config.evaluation.folds[fold_id - 1]
        ckpt_path    = ckpt_base / f"fold_{fold_id}" / "model_final.pt"
        metrics_path = results_base / f"fold_{fold_id}" / "oos_metrics.json"

        if not ckpt_path.exists():
            logger.warning(f"  Fold {fold_id}: checkpoint not found at {ckpt_path} — skipping")
            continue
        if not metrics_path.exists():
            logger.warning(f"  Fold {fold_id}: metrics file not found at {metrics_path} — skipping")
            continue

        logger.info("=" * 60)
        logger.info(
            f"Fold {fold_id}  |  OOS: {fold_spec['test_start']} – "
            f"{fold_spec.get('test_end', 'present')}"
        )

        panel_dict = _load_panel(data_dir, fold_id)
        id_mapper, num_unique_ids = build_id_mapper(panel_dict)

        ckpt = torch.load(ckpt_path, map_location=device)
        model = model_from_config(
            vars(config.architecture),
            D_g         = 20,
            num_tickers = num_unique_ids,
            num_sectors = 12,
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        logger.info(f"  Checkpoint loaded (trained to update {ckpt.get('update_step', '?')})")

        new_metrics = _oos_with_turnover(
            model=model,
            config=config,
            fold_spec=fold_spec,
            panel_dict=panel_dict,
            device=device,
            id_mapper=id_mapper,
        )
        if not new_metrics:
            continue

        turnover_mean = new_metrics.get("turnover_mean", float("nan"))
        logger.info(f"  turnover_mean = {turnover_mean:.4f} ({turnover_mean*100:.1f}% avg one-way/week)")

        # Load existing metrics, update only turnover_mean, write back
        with open(metrics_path) as f:
            existing = json.load(f)

        existing["turnover_mean"] = (
            float(turnover_mean) if not (isinstance(turnover_mean, float) and np.isnan(turnover_mean))
            else None
        )

        # Sanity-check: key OOS metrics should be identical (same checkpoint, same data)
        for key in ("excess_cagr", "sharpe", "sortino", "max_drawdown"):
            old_v = existing.get(key)
            new_v = new_metrics.get(key)
            if old_v is not None and new_v is not None:
                diff = abs(float(old_v) - float(new_v))
                if diff > 1e-4:
                    logger.warning(
                        f"  MISMATCH on {key}: stored={old_v:.6f}  recomputed={new_v:.6f}  "
                        f"diff={diff:.6f} — check for data changes"
                    )

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
        logger.info(f"  Updated {metrics_path}")

    logger.info("=" * 60)
    logger.info("Done.")


if __name__ == "__main__":
    main()
