"""
inference/checkpoint_loader.py
================================
Checkpoint loader for Project Apex inference (Bible §12.1).

Loads:
  - Model weights (ApexActorCritic state dict)
  - Normalization statistics (per-feature mean/std fitted IS-only)
  - Security ID mapping (ticker → integer id)
  - Ticker alias table (canonical → aliases)
  - Fold metadata (which fold, OOS Sortino, checkpoint step)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Checkpoint manifest dataclass
# ---------------------------------------------------------------------------

@dataclass
class CheckpointManifest:
    """Metadata associated with a saved checkpoint."""
    checkpoint_id:    str
    fold_id:          int
    update_step:      int
    oos_sortino:      float
    oos_max_drawdown: float
    oos_excess_cagr:  float
    train_end:        str             # ISO date string
    test_start:       str
    test_end:         Optional[str]
    model_config:     Dict[str, Any]  # arch hyperparams used to build model
    extra:            Dict[str, Any]  = field(default_factory=dict)


@dataclass
class LoadedCheckpoint:
    """Everything needed for inference, returned by CheckpointLoader."""
    manifest:         CheckpointManifest
    model_state_dict: Dict[str, torch.Tensor]
    norm_stats:       Dict[str, np.ndarray]   # {"mean": [F], "std": [F]}
    security_id_map:  Dict[str, int]          # ticker → integer id
    ticker_alias_map: Dict[str, List[str]]    # canonical → [aliases]
    sector_map:       Dict[str, int]          # ticker → GICS sector int

    @property
    def fold_id(self) -> int:
        return self.manifest.fold_id

    @property
    def oos_sortino(self) -> float:
        return self.manifest.oos_sortino


# ---------------------------------------------------------------------------
# CheckpointLoader
# ---------------------------------------------------------------------------

class CheckpointLoader:
    """
    §12.1  Load and validate a saved checkpoint for inference.

    Expected checkpoint directory layout::

        checkpoint_dir/
          manifest.json           # CheckpointManifest fields
          model_weights.pt        # torch state_dict
          norm_stats.npz          # mean [F], std [F]
          security_id_map.json    # {"AAPL": 0, "MSFT": 1, ...}
          ticker_alias_map.json   # {"AAPL": ["AAPL US Equity", ...]}
          sector_map.json         # {"AAPL": 0, "MSFT": 1, ...}  (GICS int)
    """

    REQUIRED_FILES = [
        "manifest.json",
        "model_weights.pt",
        "norm_stats.npz",
        "security_id_map.json",
    ]

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_directory(self) -> Tuple[bool, List[str]]:
        """
        Check that all required files are present.

        Returns
        -------
        (is_valid, missing_files)
        """
        missing = [
            f for f in self.REQUIRED_FILES
            if not (self.checkpoint_dir / f).exists()
        ]
        return len(missing) == 0, missing

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(
        self,
        device: torch.device = None,
        strict_model_load: bool = True,
    ) -> LoadedCheckpoint:
        """
        Load checkpoint from disk.

        Parameters
        ----------
        device           : torch device for model weights (default cpu)
        strict_model_load: passed to model.load_state_dict(); False allows
                           partial loads during development.

        Returns
        -------
        LoadedCheckpoint
        """
        if device is None:
            device = torch.device("cpu")

        valid, missing = self.validate_directory()
        if not valid:
            raise FileNotFoundError(
                f"Checkpoint directory '{self.checkpoint_dir}' is missing: {missing}"
            )

        # Manifest
        manifest = self._load_manifest()

        # Model weights
        model_state = torch.load(
            self.checkpoint_dir / "model_weights.pt",
            map_location=device,
        )

        # Normalization statistics
        norm_data  = np.load(self.checkpoint_dir / "norm_stats.npz")
        norm_stats = {
            "mean": norm_data["mean"].astype(np.float32),
            "std":  norm_data["std"].astype(np.float32),
        }

        # Security ID map
        with open(self.checkpoint_dir / "security_id_map.json") as f:
            security_id_map: Dict[str, int] = json.load(f)

        # Optional files
        ticker_alias_map: Dict[str, List[str]] = {}
        alias_path = self.checkpoint_dir / "ticker_alias_map.json"
        if alias_path.exists():
            with open(alias_path) as f:
                ticker_alias_map = json.load(f)

        sector_map: Dict[str, int] = {}
        sector_path = self.checkpoint_dir / "sector_map.json"
        if sector_path.exists():
            with open(sector_path) as f:
                sector_map = json.load(f)

        return LoadedCheckpoint(
            manifest         = manifest,
            model_state_dict = model_state,
            norm_stats       = norm_stats,
            security_id_map  = security_id_map,
            ticker_alias_map = ticker_alias_map,
            sector_map       = sector_map,
        )

    # ------------------------------------------------------------------
    # Saving (for checkpoint creation during training)
    # ------------------------------------------------------------------

    @classmethod
    def save(
        cls,
        checkpoint_dir:  str | Path,
        manifest:        CheckpointManifest,
        model_state_dict: Dict[str, torch.Tensor],
        norm_stats:       Dict[str, np.ndarray],
        security_id_map:  Dict[str, int],
        ticker_alias_map: Dict[str, List[str]] = None,
        sector_map:       Dict[str, int]        = None,
    ) -> None:
        """Save a checkpoint to disk."""
        out = Path(checkpoint_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Manifest
        with open(out / "manifest.json", "w") as f:
            manifest_dict = {
                "checkpoint_id":    manifest.checkpoint_id,
                "fold_id":          manifest.fold_id,
                "update_step":      manifest.update_step,
                "oos_sortino":      manifest.oos_sortino,
                "oos_max_drawdown": manifest.oos_max_drawdown,
                "oos_excess_cagr":  manifest.oos_excess_cagr,
                "train_end":        manifest.train_end,
                "test_start":       manifest.test_start,
                "test_end":         manifest.test_end,
                "model_config":     manifest.model_config,
                "extra":            manifest.extra,
            }
            json.dump(manifest_dict, f, indent=2)

        # Model weights
        torch.save(model_state_dict, out / "model_weights.pt")

        # Norm stats
        np.savez(out / "norm_stats.npz",
                 mean=norm_stats["mean"],
                 std=norm_stats["std"])

        # Security ID map
        with open(out / "security_id_map.json", "w") as f:
            json.dump(security_id_map, f, indent=2)

        # Optional
        if ticker_alias_map is not None:
            with open(out / "ticker_alias_map.json", "w") as f:
                json.dump(ticker_alias_map, f, indent=2)

        if sector_map is not None:
            with open(out / "sector_map.json", "w") as f:
                json.dump(sector_map, f, indent=2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_manifest(self) -> CheckpointManifest:
        with open(self.checkpoint_dir / "manifest.json") as f:
            d = json.load(f)
        return CheckpointManifest(
            checkpoint_id    = d["checkpoint_id"],
            fold_id          = int(d["fold_id"]),
            update_step      = int(d["update_step"]),
            oos_sortino      = float(d["oos_sortino"]),
            oos_max_drawdown = float(d["oos_max_drawdown"]),
            oos_excess_cagr  = float(d["oos_excess_cagr"]),
            train_end        = d["train_end"],
            test_start       = d["test_start"],
            test_end         = d.get("test_end"),
            model_config     = d.get("model_config", {}),
            extra            = d.get("extra", {}),
        )
