"""
training/sac_trainer.py
SAC Training Loop — Phase 9 (Bible §8.1–8.12).

Implements:
  §8.1.1  Critic loss: QR-Huber with distributional Bellman targets
  §8.1.2  Actor loss: reparameterization in logit space, pre-projection entropy
  §8.1.3  Alpha loss: auto-temperature tuning (K-dependent H_target)
  §8.7    Update schedule: critic every step; actor+alpha every policy_delay steps
  §8.8    Entropy target: H_target = −ln(K_active) × scale_factor
  §8.9    Optimizer config: separate AdamW/Adam per component; embeddings wd=0
  §8.10   Polyak averaging: τ_Q = 0.005, applied after EVERY critic step
  §8.11   Gradient clipping: critic=1.0, encoder=1.0, actor=5.0

Observation reconstruction:
  Observations (x_t, g_t) are NOT stored in the replay buffer; they are
  reconstructed from t_idx at sample time using the panel arrays passed to
  SACTrainer at construction.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from model.apex_actor_critic import ApexActorCritic

EPS = 1e-8


# ---------------------------------------------------------------------------
# Standalone utilities (also used by tests directly)
# ---------------------------------------------------------------------------

def polyak_update(
    online: nn.Module,
    target: nn.Module,
    tau: float,
) -> None:
    """
    In-place Polyak average (§8.10):
      θ_target ← τ · θ_online + (1 − τ) · θ_target

    Covers ALL parameters of the target model (actor params are present but
    never used via target.critic_forward(), so their values are irrelevant).
    """
    with torch.no_grad():
        for p_on, p_targ in zip(online.parameters(), target.parameters()):
            p_targ.data.mul_(1.0 - tau)
            p_targ.data.add_(tau * p_on.data)


def qr_huber_loss(
    q_pred:   torch.Tensor,   # [B, N_q]  online predicted quantiles
    z_target: torch.Tensor,   # [B, N_q]  distributional target (detached)
    taus:     torch.Tensor,   # [N_q]     fixed quantile levels
    delta:    float = 1.0,
) -> torch.Tensor:
    """
    Quantile-Huber (QR-Huber) loss (§8.1.1).

    Full distributional target: both z_j and q̂_k are quantile vectors.

      L = (1/N²) Σ_j Σ_k  |τ_k − 1(z_j − q̂_k < 0)| · L_δ(z_j − q̂_k)

    where j indexes target quantiles, k indexes predicted quantiles, and
    τ_k is the fixed quantile level for the k-th predicted quantile.

    Shapes:
        q_pred   [B, N]
        z_target [B, N]
        taus     [N]
    Returns scalar.
    """
    B, N = q_pred.shape
    # Pairwise differences: z_j − q̂_k → [B, N_j, N_k]
    u = z_target.unsqueeze(2) - q_pred.unsqueeze(1)   # [B, N_j, N_k]

    # Huber component: L_δ(u) = 0.5·u² if |u|<δ  else  δ·(|u| − 0.5δ)
    abs_u = u.abs()
    huber = torch.where(abs_u < delta,
                        0.5 * u * u,
                        delta * (abs_u - 0.5 * delta))

    # Asymmetric quantile weight: |τ_j − 1(u < 0)|  (§8.1.1 bible convention)
    # τ_j is the quantile level for the target dimension j → broadcast over k
    tau_j     = taus.view(1, N, 1)               # [1, N_j, 1]
    indicator = (u < 0).float()                  # [B, N_j, N_k]
    weight    = (tau_j - indicator).abs()        # [B, N_j, N_k]

    # Mean over (B, N_j, N_k)  — equivalent to (1/N²) average over quantile pairs
    return (weight * huber).mean()


# ---------------------------------------------------------------------------
# SACTrainer
# ---------------------------------------------------------------------------

class SACTrainer:
    """
    SAC training loop for Project Apex (Bible §8.1–8.12).

    Parameters
    ----------
    model       : Online ApexActorCritic (modified in-place during training)
    panel_data  : dict with numpy arrays for obs reconstruction:
                    x_panel    [T, K, F]  float32
                    g_panel    [T, D_g]   float32
                    mask_panel [T, K]     float32
                    ticker_ids [T, K]     int64  (security ids; −1 = inactive)
                    sector_ids [T, K]     int64  (GICS sector code; −1 = inactive)
    sac_cfg     : SACConfig (or any object with matching attributes)
    opt_cfg     : OptimizerConfig (or any object with matching attributes)
    L_lookback  : panel-row lookback window (default 60, matching env default)
    device      : torch device (defaults to cpu)
    """

    def __init__(
        self,
        model:      ApexActorCritic,
        panel_data: Dict[str, np.ndarray],
        sac_cfg,
        opt_cfg,
        L_lookback: int           = 60,
        device:     torch.device  = None,
        id_mapper:  Optional[Dict[int, int]] = None,
    ):
        if device is None:
            device = torch.device("cpu")
        self.device = device
        self._id_mapper = id_mapper
        
        # Build vectorized ID mapping lookup array (cached)
        self._id_lookup = None
        if id_mapper is not None:
            max_id = max(id_mapper.keys())
            self._id_lookup = np.zeros(max_id + 1, dtype=np.int64)
            for sparse_id, dense_idx in id_mapper.items():
                self._id_lookup[sparse_id] = dense_idx

        # Online model
        self.model = model.to(device)
        self.model.train()

        # §8.10 Target model — deep copy, gradient-free
        self.model_target = copy.deepcopy(model).to(device)
        for p in self.model_target.parameters():
            p.requires_grad_(False)
        self.model_target.eval()

        # Panel arrays (kept as numpy; sliced per-batch and converted to tensors)
        self._x_panel    = np.asarray(panel_data["x_panel"],    dtype=np.float32)
        self._g_panel    = np.asarray(panel_data["g_panel"],    dtype=np.float32)
        self._mask_panel = np.asarray(panel_data["mask_panel"], dtype=np.float32)
        self._ticker_ids = np.asarray(panel_data["ticker_ids"], dtype=np.int64)
        self._sector_ids = np.asarray(panel_data["sector_ids"], dtype=np.int64)
        self._L          = int(L_lookback)

        # SAC hyperparameters
        self._tau              = float(sac_cfg.tau)
        self._policy_delay     = int(sac_cfg.policy_delay)
        self._clip_critic      = float(sac_cfg.grad_clip_critic)
        self._clip_actor       = float(sac_cfg.grad_clip_actor)
        self._clip_encoder     = float(sac_cfg.grad_clip_encoder)
        self._entropy_scale    = float(sac_cfg.entropy_scale_factor)
        self._alpha_clamp_min  = float(sac_cfg.alpha_clamp_min)
        self._alpha_clamp_max  = float(sac_cfg.alpha_clamp_max)

        # §8.8 log_alpha (optimised in log-space; α = exp(log_α))
        init_log_alpha = math.log(float(sac_cfg.init_alpha))
        self.log_alpha = nn.Parameter(
            torch.tensor(init_log_alpha, dtype=torch.float32, device=device)
        )

        # §8.9 Optimizers
        self._build_optimizers(sac_cfg, opt_cfg)

        # Step counter (drives policy_delay logic)
        self._update_step = 0

        # §8.12 Stability diagnostic counters
        self._entropy_collapse_steps = 0   # consecutive steps with entropy < 0.01
        self._alpha_pinned_steps     = 0   # consecutive steps with alpha ≈ alpha_max

        # Last computed metrics (for external inspection / tests)
        self.last_metrics: Dict = {}

    # ======================================================================
    # Properties
    # ======================================================================

    @property
    def alpha(self) -> float:
        """Current temperature α = exp(log_α)."""
        return float(self.log_alpha.exp().item())

    @property
    def update_step(self) -> int:
        return self._update_step

    # ======================================================================
    # Main update step  (§8.7)
    # ======================================================================

    def update(
        self,
        batch: Dict,
        rng:   Optional[np.random.Generator] = None,
    ) -> Dict:
        """
        Run one SAC update (§8.7).

        Parameters
        ----------
        batch : dict from ReplayBuffer.sample(critic=True) with keys:
                  t_idx, t_idx_next, mask_t, w_pre,
                  R_n, gamma_n, done_n, aug_obs_std_factor
        rng   : numpy RNG for obs augmentation (None → no augmentation)

        Returns
        -------
        metrics dict with all diagnostic scalars.
        """
        self._update_step += 1
        metrics: Dict = {}

        # Build tensor observations for current + next states
        (x_t,  g_t,  mask_t,  tid_t,  sid_t,
         x_tn, g_tn, mask_tn, tid_tn, sid_tn,
         R_n, gamma_n, done_n, w_pre_buf) = self._prepare_batch(batch, rng)

        # ------------------------------------------------------------------
        # §8.1.1  Critic update (every step)
        # ------------------------------------------------------------------
        c_metrics = self._critic_step(
            x_t,  g_t,  mask_t,  tid_t,  sid_t,
            x_tn, g_tn, mask_tn, tid_tn, sid_tn,
            R_n, gamma_n, done_n, w_pre_buf,
        )
        metrics.update(c_metrics)

        # ------------------------------------------------------------------
        # §8.1.2 / §8.1.3  Actor + Alpha update (every policy_delay steps)
        # ------------------------------------------------------------------
        if self._update_step % self._policy_delay == 0:
            a_metrics = self._actor_alpha_step(
                x_t, g_t, mask_t, tid_t, sid_t,
            )
            metrics.update(a_metrics)
        else:
            metrics["actor_loss"]  = float("nan")
            metrics["alpha_loss"]  = float("nan")
            metrics["entropy_mean"] = float("nan")

        # ------------------------------------------------------------------
        # §8.10  Polyak update of target critics (after EVERY critic step)
        # ------------------------------------------------------------------
        polyak_update(self.model, self.model_target, self._tau)

        metrics["alpha"]       = self.alpha
        metrics["log_alpha"]   = float(self.log_alpha.item())
        metrics["update_step"] = self._update_step

        # ------------------------------------------------------------------
        # §8.12  Stability diagnostics / alarms (Table 51)
        # ------------------------------------------------------------------
        q1_abs = abs(metrics.get("q1_mean", 0.0))
        q2_abs = abs(metrics.get("q2_mean", 0.0))
        metrics["q_divergence_flag"] = bool(q1_abs > 100.0 or q2_abs > 100.0)

        ent = metrics.get("entropy_mean", float("nan"))
        if not math.isnan(ent):
            self._entropy_collapse_steps = (
                self._entropy_collapse_steps + 1 if ent < 0.01 else 0
            )
        metrics["entropy_collapse_flag"] = self._entropy_collapse_steps > 100

        alpha_max_val = math.exp(self._alpha_clamp_max)
        if abs(self.alpha - alpha_max_val) < 1e-4:
            self._alpha_pinned_steps += 1
        else:
            self._alpha_pinned_steps = 0
        metrics["alpha_pinned_max_flag"] = self._alpha_pinned_steps > 200

        self.last_metrics = metrics
        return metrics

    # ======================================================================
    # Optimizer construction  (§8.9)
    # ======================================================================

    def _build_optimizers(self, sac_cfg, opt_cfg) -> None:
        """Build 4 separate optimizers: encoder (AdamW), actor, critic, alpha (Adam)."""

        # Identify embedding parameter ids (receive wd=0)
        emb_ids = {id(p) for p in self.model.embedding_parameters()}

        # --- Encoder param groups ---
        # Non-embedding encoder params: tcn, emb_proj, global_proj  → weight decay
        # Embedding params: ticker_emb, sector_emb               → wd = 0
        enc_wd_params   = []
        enc_no_wd_params = list(self.model.embedding_parameters())

        _enc_prefixes = ("tcn.", "emb_proj.", "global_proj.")
        for name, p in self.model.named_parameters():
            if id(p) not in emb_ids:
                if any(name.startswith(pfx) for pfx in _enc_prefixes):
                    enc_wd_params.append(p)

        self._enc_wd = float(opt_cfg.encoder_weight_decay)
        self.enc_optimizer = torch.optim.AdamW(
            [
                {"params": enc_wd_params,    "weight_decay": self._enc_wd},
                {"params": enc_no_wd_params, "weight_decay": 0.0},
            ],
            lr=float(opt_cfg.encoder_lr),
        )

        # --- Actor params ---
        _actor_prefixes = ("actor_attn.", "actor_pool.", "actor_mu_head.")
        actor_params = [
            p for name, p in self.model.named_parameters()
            if any(name.startswith(pfx) for pfx in _actor_prefixes)
            or name == "log_sigma"
        ]
        self.actor_optimizer = torch.optim.Adam(
            actor_params,
            lr=float(opt_cfg.actor_lr),
            weight_decay=float(opt_cfg.actor_weight_decay),
        )

        # --- Critic params ---
        _critic_prefixes = ("q1_attn.", "q1_pool.", "q1_head.",
                            "q2_attn.", "q2_pool.", "q2_head.")
        critic_params = [
            p for name, p in self.model.named_parameters()
            if any(name.startswith(pfx) for pfx in _critic_prefixes)
        ]
        self.critic_optimizer = torch.optim.Adam(
            critic_params,
            lr=float(opt_cfg.critic_lr),
            weight_decay=float(opt_cfg.critic_weight_decay),
        )

        # --- Alpha ---
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=float(sac_cfg.alpha_lr),
        )

    # ======================================================================
    # Observation reconstruction
    # ======================================================================

    def _reconstruct_obs(
        self,
        t_idx_arr:        np.ndarray,      # [B] int64
        aug_noise_factor: float = 0.0,
        rng:              Optional[np.random.Generator] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Rebuild (x, g, mask, ticker_ids, sector_ids) tensors for a batch of
        panel-row indices.

        The lookback window contains exactly L timesteps with t as the final
        (most recent) step, matching the model's expected input [B, L, K, F]:
          p_start = max(0, t_idx + 1 - L)
          x_win   = x_panel[p_start : t_idx + 1]   # length L when t_idx >= L-1

        Returns
        -------
        x         [B, L, K, F]
        g         [B, D_g]
        mask      [B, K]
        ticker_ids [B, K]  int64
        sector_ids [B, K]  int64
        """
        B       = len(t_idx_arr)
        L       = self._L
        K       = self._x_panel.shape[1]
        F       = self._x_panel.shape[2]
        D_g     = self._g_panel.shape[1]
        win_len = L         # exactly L timesteps; t is the final step

        x_batch = np.zeros((B, win_len, K, F), dtype=np.float32)
        g_batch = np.zeros((B, D_g),           dtype=np.float32)

        for i, tidx in enumerate(t_idx_arr):
            tidx = int(tidx)
            p_start = max(0, tidx + 1 - win_len)
            win = self._x_panel[p_start : tidx + 1]   # [<=L, K, F]
            wl  = win.shape[0]
            x_batch[i, -wl:] = win      # right-align in fixed [L+1, K, F] buffer

            # §8.4 obs augmentation (applied only when aug_noise_factor > 0)
            if aug_noise_factor > 0.0 and rng is not None:
                noise = rng.normal(
                    0.0, aug_noise_factor,
                    size=x_batch[i].shape
                ).astype(np.float32)
                x_batch[i] += noise

            g_batch[i] = self._g_panel[tidx]

        mask_batch = self._mask_panel[t_idx_arr]    # [B, K]
        tid_batch  = self._ticker_ids[t_idx_arr]    # [B, K]
        sid_batch  = self._sector_ids[t_idx_arr]    # [B, K]

        # Apply ID mapping if provided (fully vectorized using cached lookup)
        if self._id_lookup is not None:
            # Vectorized mapping: clip negative IDs to 0, then lookup
            max_id = len(self._id_lookup) - 1
            tid_batch_clipped = np.clip(tid_batch, 0, max_id)
            tid_batch = self._id_lookup[tid_batch_clipped]

        x   = torch.from_numpy(x_batch).to(self.device)
        g   = torch.from_numpy(g_batch).to(self.device)
        m   = torch.from_numpy(mask_batch).to(self.device)
        tid = torch.from_numpy(tid_batch).long().to(self.device)
        sid = torch.from_numpy(sid_batch).long().to(self.device)

        return x, g, m, tid, sid

    def _prepare_batch(
        self,
        batch: Dict,
        rng:   Optional[np.random.Generator],
    ):
        """Convert buffer batch dict to tensors for current + next states."""
        t_arr  = batch["t_idx"]                     # [B] int64
        tn_arr = batch["t_idx_next"]                # [B] int64
        aug    = float(batch.get("aug_obs_std_factor", 0.0))

        # Current state (augmented for critic path per §8.4)
        x_t, g_t, mask_t, tid_t, sid_t = self._reconstruct_obs(t_arr, aug, rng)

        # Next state (clean; no augmentation on bootstrap target)
        x_tn, g_tn, mask_tn, tid_tn, sid_tn = self._reconstruct_obs(tn_arr, 0.0, None)

        R_n     = torch.from_numpy(batch["R_n"].astype(np.float32)).to(self.device)
        gamma_n = torch.from_numpy(batch["gamma_n"].astype(np.float32)).to(self.device)
        done_n  = torch.from_numpy(batch["done_n"].astype(np.float32)).to(self.device)
        w_pre   = torch.from_numpy(batch["w_pre"].astype(np.float32)).to(self.device)

        return (x_t,  g_t,  mask_t,  tid_t,  sid_t,
                x_tn, g_tn, mask_tn, tid_tn, sid_tn,
                R_n, gamma_n, done_n, w_pre)

    # ======================================================================
    # Critic update step  (§8.1.1)
    # ======================================================================

    def _critic_step(
        self,
        x_t,  g_t,  mask_t,  tid_t,  sid_t,
        x_tn, g_tn, mask_tn, tid_tn, sid_tn,
        R_n, gamma_n, done_n, w_pre_buf,
    ) -> Dict:
        """Compute QR-Huber critic loss, backward, clip, step."""
        # ------------------------------------------------------------------
        # §8.1.1  Distributional Bellman target  (no gradient)
        # ------------------------------------------------------------------
        with torch.no_grad():
            self.model.train()   # ensure noise active for next-state actor sampling

            # Next action from ONLINE actor (reparameterization trick, §8.1.2)
            w_pre_next, log_prob_next = self.model.actor_forward(
                x_tn, g_tn, mask_tn, sid_tn, tid_tn
            )   # [B, K], [B]  (sector_ids before ticker_ids)

            # Target critics at next state (frozen Polyak-averaged params)
            q1_targ, q2_targ = self.model_target.critic_forward(
                x_tn, g_tn, mask_tn, sid_tn, tid_tn, w_pre_next
            )   # [B, N_q] each

            # Element-wise min of target distributions (twin-critic aggregation)
            q_min_targ = torch.min(q1_targ, q2_targ)   # [B, N_q]

            # V_soft: entropy-corrected soft value  (§8.1.1)
            alpha = self.log_alpha.exp().detach()
            v_soft = q_min_targ - alpha * log_prob_next.unsqueeze(1)  # [B, N_q]

            # Distributional Bellman target z_j = R_n + γ^n·(1−done_n)·V_soft_j
            z_target = (
                R_n.unsqueeze(1)
                + gamma_n.unsqueeze(1) * (1.0 - done_n.unsqueeze(1)) * v_soft
            )   # [B, N_q]

        # ------------------------------------------------------------------
        # Online critics at current state
        # ------------------------------------------------------------------
        q1_pred, q2_pred = self.model.critic_forward(
            x_t, g_t, mask_t, sid_t, tid_t, w_pre_buf
        )   # [B, N_q] each

        taus = self.model.taus   # [N_q]
        loss_q1 = qr_huber_loss(q1_pred, z_target, taus)
        loss_q2 = qr_huber_loss(q2_pred, z_target, taus)
        critic_loss = loss_q1 + loss_q2

        # ------------------------------------------------------------------
        # Backward → clip → step
        # ------------------------------------------------------------------
        self.enc_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        enc_gnorm_pre    = _grad_norm(self.enc_optimizer)
        critic_gnorm_pre = _grad_norm(self.critic_optimizer)

        nn.utils.clip_grad_norm_(
            _opt_params(self.enc_optimizer),    self._clip_encoder
        )
        nn.utils.clip_grad_norm_(
            _opt_params(self.critic_optimizer), self._clip_critic
        )

        enc_gnorm_post    = _grad_norm(self.enc_optimizer)
        critic_gnorm_post = _grad_norm(self.critic_optimizer)

        self.enc_optimizer.step()
        self.critic_optimizer.step()

        return {
            "critic_loss":           float(critic_loss.item()),
            "loss_q1":               float(loss_q1.item()),
            "loss_q2":               float(loss_q2.item()),
            "q1_mean":               float(q1_pred.mean().item()),
            "q2_mean":               float(q2_pred.mean().item()),
            "q_target_mean":         float(z_target.mean().item()),
            "td_error_abs_mean":     float((z_target - q1_pred).abs().mean().item()),
            "enc_grad_norm_crit":    enc_gnorm_pre,
            "critic_grad_norm":      critic_gnorm_pre,
            "enc_grad_norm_crit_post":    enc_gnorm_post,
            "critic_grad_norm_post":      critic_gnorm_post,
            "critic_grad_clipped":   bool(critic_gnorm_pre > self._clip_critic),
            "enc_grad_clipped_crit": bool(enc_gnorm_pre    > self._clip_encoder),
        }

    # ======================================================================
    # Actor + Alpha update step  (§8.1.2, §8.1.3)
    # ======================================================================

    def _actor_alpha_step(
        self,
        x_t, g_t, mask_t, tid_t, sid_t,
    ) -> Dict:
        """Compute actor loss + alpha loss, backward, clip, step."""

        # ------------------------------------------------------------------
        # Actor: reparameterization in logit space (§8.1.2)
        # ------------------------------------------------------------------
        self.model.train()
        w_pre_new, log_prob = self.model.actor_forward(
            x_t, g_t, mask_t, sid_t, tid_t
        )   # [B, K], [B]  (sector_ids before ticker_ids)

        # Online critics evaluate the new action  (gradients flow through actor)
        q1_actor, q2_actor = self.model.critic_forward(
            x_t, g_t, mask_t, sid_t, tid_t, w_pre_new
        )   # [B, N_q] each

        # Expected Q: mean over quantile dimension  →  [B]
        min_q = torch.min(q1_actor.mean(dim=1), q2_actor.mean(dim=1))  # [B]

        alpha_detached = self.log_alpha.exp().detach()

        # §8.1.2 Actor loss  (minimise α·log_π − min_Q)
        actor_loss = (alpha_detached * log_prob - min_q).mean()

        # ------------------------------------------------------------------
        # §8.8 K-dependent entropy target
        # ------------------------------------------------------------------
        K_active = mask_t.sum(dim=1).clamp(min=1.0)           # [B]
        H_target = -torch.log(K_active) * self._entropy_scale  # [B]

        # §8.1.3 Alpha loss  (optimise log_α, entropy feedback)
        alpha_loss = -(
            self.log_alpha * (log_prob.detach() + H_target.detach())
        ).mean()

        # ------------------------------------------------------------------
        # Actor backward → clip → step encoder + actor
        # ------------------------------------------------------------------
        self.enc_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        actor_gnorm_pre = _grad_norm(self.actor_optimizer)
        enc_gnorm_pre   = _grad_norm(self.enc_optimizer)

        nn.utils.clip_grad_norm_(
            _opt_params(self.actor_optimizer), self._clip_actor
        )
        nn.utils.clip_grad_norm_(
            _opt_params(self.enc_optimizer),   self._clip_encoder
        )

        actor_gnorm_post = _grad_norm(self.actor_optimizer)
        enc_gnorm_post   = _grad_norm(self.enc_optimizer)

        self.actor_optimizer.step()
        self.enc_optimizer.step()

        # ------------------------------------------------------------------
        # Alpha backward → step → clamp
        # ------------------------------------------------------------------
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # §8.1.3 Clamp log_α to [log(α_min), log(α_max)]
        with torch.no_grad():
            self.log_alpha.clamp_(self._alpha_clamp_min, self._alpha_clamp_max)

        entropy_mean  = float(-log_prob.mean().item())
        H_target_mean = float(H_target.mean().item())

        return {
            "actor_loss":             float(actor_loss.item()),
            "alpha_loss":             float(alpha_loss.item()),
            "entropy_mean":           entropy_mean,
            "H_target_mean":          H_target_mean,
            "entropy_gap":            entropy_mean - H_target_mean,
            "actor_grad_norm":        actor_gnorm_pre,
            "enc_grad_norm_act":      enc_gnorm_pre,
            "actor_grad_norm_post":   actor_gnorm_post,
            "enc_grad_norm_act_post": enc_gnorm_post,
            "actor_grad_clipped":     bool(actor_gnorm_pre > self._clip_actor),
            "enc_grad_clipped_act":   bool(enc_gnorm_pre   > self._clip_encoder),
        }

    # ======================================================================
    # Config-driven constructor
    # ======================================================================

    @classmethod
    def from_config(
        cls,
        model:              ApexActorCritic,
        panel_data:         Dict[str, np.ndarray],
        cfg,                # ProjectConfig
        device:             torch.device = None,
        use_deterministic:  bool         = False,
    ) -> "SACTrainer":
        """
        Construct SACTrainer from a ProjectConfig.

        Parameters
        ----------
        use_deterministic : bool
            If True, enforce torch.use_deterministic_algorithms(True) per §8.9
            (CUDA deterministic ops). Disable only if a required op lacks a
            deterministic implementation on the target device.
        """
        if use_deterministic:
            torch.use_deterministic_algorithms(True)
        return cls(
            model      = model,
            panel_data = panel_data,
            sac_cfg    = cfg.sac,
            opt_cfg    = cfg.optimizer,
            L_lookback = cfg.architecture.L,
            device     = device,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _opt_params(optimizer: torch.optim.Optimizer) -> List[nn.Parameter]:
    """Flatten all parameters from an optimizer's param groups."""
    params = []
    for group in optimizer.param_groups:
        params.extend(group["params"])
    return params


def _grad_norm(optimizer: torch.optim.Optimizer) -> float:
    """L2 gradient norm across all params in the optimizer."""
    total = 0.0
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total)
