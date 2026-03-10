"""
Generate Project Apex Audit Report — Bible v5 vs Implementation
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether,
)
from reportlab.platypus.flowables import Flowable
import datetime


# ---------------------------------------------------------------------------
# Custom styles
# ---------------------------------------------------------------------------

def build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="CoverTitle",
        fontName="Helvetica-Bold",
        fontSize=28,
        leading=34,
        alignment=TA_CENTER,
        textColor=HexColor("#1a1a2e"),
        spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        name="CoverSub",
        fontName="Helvetica",
        fontSize=14,
        leading=18,
        alignment=TA_CENTER,
        textColor=HexColor("#555555"),
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="SectionH1",
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=24,
        textColor=HexColor("#1a1a2e"),
        spaceBefore=24,
        spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        name="SectionH2",
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=HexColor("#2d3436"),
        spaceBefore=16,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="SectionH3",
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=HexColor("#2d3436"),
        spaceBefore=10,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="BodyJ",
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="ApexBullet",
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        leftIndent=18,
        bulletIndent=6,
        spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        name="ApexBulletBold",
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        leftIndent=18,
        bulletIndent=6,
        spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        name="StatusGood",
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=HexColor("#27ae60"),
    ))
    styles.add(ParagraphStyle(
        name="StatusBad",
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=HexColor("#c0392b"),
    ))
    styles.add(ParagraphStyle(
        name="StatusWarn",
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=HexColor("#e67e22"),
    ))
    styles.add(ParagraphStyle(
        name="SmallNote",
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=10,
        textColor=HexColor("#888888"),
        spaceAfter=4,
    ))
    return styles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hr():
    return HRFlowable(width="100%", thickness=1, color=HexColor("#cccccc"),
                       spaceBefore=6, spaceAfter=6)

def severity_tag(sev):
    colors = {"CRITICAL": "#c0392b", "HIGH": "#e67e22", "MEDIUM": "#f1c40f", "LOW": "#27ae60"}
    c = colors.get(sev, "#555555")
    return f'<font color="{c}"><b>[{sev}]</b></font>'

def status_table(rows, col_widths=None):
    """rows: list of [Component, Bible, Implementation, Status] lists"""
    if col_widths is None:
        col_widths = [1.8*inch, 1.6*inch, 2.0*inch, 1.4*inch]
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 9),
        ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 1), (-1, -1), 8),
        ("LEADING",    (0, 0), (-1, -1), 11),
        ("ALIGN",      (0, 0), (-1, -1), "LEFT"),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ("GRID",       (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f7f9fc")]),
        ("LEFTPADDING",  (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
    ]))
    return t


# ---------------------------------------------------------------------------
# Report content
# ---------------------------------------------------------------------------

def build_report(output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        topMargin=0.6*inch,
        bottomMargin=0.6*inch,
        leftMargin=0.7*inch,
        rightMargin=0.7*inch,
    )
    S = build_styles()
    story = []

    # ===== COVER =====
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph("PROJECT APEX", S["CoverTitle"]))
    story.append(Paragraph("Comprehensive Audit Report", S["CoverSub"]))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Bible v5 vs. Implementation Gap Analysis", S["CoverSub"]))
    story.append(Paragraph("Code Quality Review &amp; Pre-Run Readiness Assessment", S["CoverSub"]))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')}", S["CoverSub"]))
    story.append(Paragraph("Confidential — Internal Use Only", S["SmallNote"]))
    story.append(PageBreak())

    # ===== TABLE OF CONTENTS =====
    story.append(Paragraph("Table of Contents", S["SectionH1"]))
    toc_items = [
        "1. Executive Summary",
        "2. Architecture Alignment — Bible vs. Implementation",
        "3. Hyperparameter Audit",
        "4. Feature Engineering Gap Analysis",
        "5. Environment &amp; Reward Function",
        "6. Model Architecture Compliance",
        "7. Training Pipeline (SAC / Replay / Warmup)",
        "8. Evaluation Framework",
        "9. Inference &amp; Paper Trading",
        "10. Logging &amp; Observability",
        "11. Code Bugs &amp; Issues Found",
        "12. Data Pipeline Concerns",
        "13. Recommendations — Priority Ranked",
        "14. Pre-Run Checklist",
    ]
    for item in toc_items:
        story.append(Paragraph(item, S["BodyJ"]))
    story.append(PageBreak())

    # ===== 1. EXECUTIVE SUMMARY =====
    story.append(Paragraph("1. Executive Summary", S["SectionH1"]))
    story.append(hr())
    story.append(Paragraph(
        "This report compares every component of the Project Apex codebase against the canonical "
        "engineering reference (Bible v5). The project is a <b>Soft Actor-Critic (SAC) reinforcement "
        "learning agent</b> designed to manage a long-only equity portfolio drawn from the Nasdaq-100 "
        "universe, rebalancing weekly and targeting risk-adjusted excess return over QQQ.",
        S["BodyJ"],
    ))
    story.append(Paragraph(
        "<b>Overall Assessment:</b> The project is substantially implemented through approximately "
        "<b>Phase 9 of 13</b> planned phases. Core components (features, environment, model, reward, "
        "replay buffer, SAC trainer) are present and architecturally aligned with the Bible. "
        "However, several gaps remain before the system is fully runnable end-to-end.",
        S["BodyJ"],
    ))

    story.append(Paragraph("Key Findings", S["SectionH2"]))

    findings = [
        ("<b>Feature Count Mismatch (F=25 vs F=26):</b> The features/__init__.py declares 18 TS features "
         "(including adj_close) and F_TOTAL=26, but feature_panel.py and per_asset_features.py produce "
         "only 17 TS features (no adj_close), giving F=25. Config says F=25. This inconsistency will "
         "cause dimension errors at import time or runtime if the wrong constant is used."),
        ("<b>Portfolio-State Features are STUB zeros:</b> All 8 portfolio-state features in g_t are "
         "hardcoded zeros. This means the agent receives no information about its own portfolio state "
         "(turnover, drawdown, vol, cost estimates). This significantly degrades agent performance."),
        ("<b>Fold 1 train_start discrepancy:</b> Bible Table says 2004-01-01, config says 2005-01-01."),
        ("<b>Parquet data files are placeholder stubs:</b> daily_bars.parquet (133 bytes), "
         "macro_features.parquet (132 bytes), and other parquet files appear to be empty stubs. "
         "The system cannot run without real data."),
        ("<b>Shared encoder architecture diverges from Bible §7.2:</b> Bible says TCN + Attention are "
         "shared between actor and critic. Implementation uses shared TCN but THREE separate attention "
         "stacks (actor_attn, q1_attn, q2_attn). This is a significant architectural deviation."),
        ("<b>NAV evolution uses previous weights, not new weights:</b> In trading_env.py line 263, "
         "portfolio return is computed using self._w_exec (previous weights) instead of the newly "
         "computed w_exec. This may be intentional (drift model) but differs from the Bible §5.3 "
         "formula which uses w_exec_t (the new weights)."),
        ("<b>Log-sigma is a shared scalar, not per-asset:</b> Bible §4.4 says the actor outputs "
         "per-asset log_std. Implementation uses a single shared scalar log_sigma for all assets."),
    ]
    for f in findings:
        story.append(Paragraph(f, S["ApexBullet"], bulletText="\u2022"))

    story.append(PageBreak())

    # ===== 2. ARCHITECTURE ALIGNMENT =====
    story.append(Paragraph("2. Architecture Alignment — Bible vs. Implementation", S["SectionH1"]))
    story.append(hr())

    arch_rows = [
        ["Component", "Bible Spec", "Implementation", "Status"],
        ["Data Pipeline\n(§2)", "daily_bars, ndx_membership,\nmacro_features, calendar,\nticker_alias",
         "All file references present.\nParquet files are stubs (133B).\nCSV files exist with real data.",
         "PARTIAL\n(data stubs)"],
        ["Feature Panel\n(§3 / §4.1)", "[T, K_max, F] panel\nF=26 (Bible §0.2)\nF=25 (config)",
         "FeaturePanelBuilder produces\n[T, K_max, 25]. __init__.py\nsays F=26 (18+8).",
         "MISMATCH\n(F inconsistency)"],
        ["Per-Asset TS\n(§3.1)", "17 features listed\nin Bible Table",
         "17 features implemented\nin per_asset_features.py",
         "MATCH"],
        ["Cross-Sectional\n(§3.2)", "8 features listed",
         "8 features implemented\nin cross_sectional_features.py",
         "MATCH"],
        ["Macro Features\n(§3.3)", "9 instruments → 9 feats\nin g_t",
         "9 features in\nmacro_broadcast_features.py",
         "MATCH"],
        ["Benchmark Feats\n(§3.5)", "3 QQQ features in g_t",
         "3 features in\nbenchmark_features.py",
         "MATCH"],
        ["Portfolio-State\n(§3.4)", "8 features in g_t\n(turnover, vol, DD, etc.)",
         "STUB: all zeros.\nNeeds environment loop.",
         "NOT IMPL"],
        ["Normalization\n(§3.6)", "Causal per-asset (52w)\nFixed-scale macro\nClip ±4",
         "CausalPerAssetNormalizer\nFixedScaleNormalizer\nClip=4.0",
         "MATCH"],
        ["Constraint\nProjector (§4.5)", "Differentiable Dykstra's\nper_name=0.15, sector=0.35",
         "Implemented in PyTorch\nDykstra's alternating proj.\nmax_iters=200",
         "MATCH"],
        ["Trading Env\n(§5)", "NAV evolution, cost model,\nforced liquidation, QQQ",
         "Full implementation\nin trading_env.py",
         "MATCH"],
        ["Reward Fn\n(§6)", "5-term formula\nwith cold-start §6.4",
         "Implemented in reward_fn.py\nwith cold-start blending",
         "MATCH"],
        ["TCN Encoder\n(§7.4)", "5 levels, k=3, 128ch\nSiLU, LayerNorm, causal",
         "CausalTCN with residual\nblocks, Pre-LN, SiLU",
         "MATCH"],
        ["Cross-Asset Attn\n(§7.5)", "SHARED attn stack\n(Bible §7.2 Table)",
         "THREE separate stacks\n(actor_attn, q1_attn, q2_attn)",
         "DIVERGENCE"],
        ["Actor Head\n(§7.7)", "Per-asset MLP [128,128]\nPer-asset log_std",
         "MLP [128,128] ✓\nShared scalar log_sigma",
         "PARTIAL\n(log_std mismatch)"],
        ["Critic Head\n(§7.8)", "MLP [256,256]\n32 quantiles, twin",
         "Twin Q1/Q2 heads\n[256,256], 32 quantiles",
         "MATCH"],
        ["Replay Buffer\n(§8.3)", "Cap=800, recency weight\nn-step=4, warmup drop",
         "Circular buffer, all\nfeatures implemented",
         "MATCH"],
        ["SAC Trainer\n(§8)", "Critic+Actor+Alpha\nPolyak, grad clip, etc.",
         "Full implementation\nin sac_trainer.py",
         "MATCH"],
        ["Walk-Forward\n(§9.1)", "8 folds, embargo=4\nexpanding window",
         "FoldManager with 8 folds.\nFold 1 start: 2005 vs 2004",
         "PARTIAL\n(fold 1 date)"],
        ["Leakage Suite\n(§9.2 / §11.3)", "5 leakage trap tests",
         "LeakageSuite with all 5\ntests implemented",
         "MATCH"],
        ["Bootstrap CI\n(§9.5)", "Moving-block bootstrap\n10K resamples",
         "Implemented in\nbootstrap.py",
         "MATCH"],
        ["Inference\n(§12)", "Guardrails, missing data,\npaper trade loop",
         "Full inference package\nwith alerts and guardrails",
         "MATCH"],
        ["Logging\n(§10)", "7 categories, 4 cadences\n7 regression alarms",
         "ApexLogger with all\ncategories and alarms",
         "MATCH"],
    ]

    story.append(status_table(arch_rows, [1.2*inch, 1.6*inch, 1.8*inch, 1.1*inch]))
    story.append(PageBreak())

    # ===== 3. HYPERPARAMETER AUDIT =====
    story.append(Paragraph("3. Hyperparameter Audit", S["SectionH1"]))
    story.append(hr())
    story.append(Paragraph(
        "All hyperparameters from Bible §0.2 Master Hyperparameter Table were compared against "
        "master_config.yaml values:",
        S["BodyJ"],
    ))

    hp_rows = [
        ["Parameter", "Bible Default", "Config Value", "Match?"],
        ["gamma", "0.975", "0.975", "YES"],
        ["n_step", "4", "4", "YES"],
        ["updates_per_step", "20", "20", "YES"],
        ["policy_delay", "2", "2", "YES"],
        ["batch_size", "64", "64", "YES"],
        ["tau (Polyak)", "0.005", "0.005", "YES"],
        ["grad_clip_critic", "1.0", "1.0", "YES"],
        ["grad_clip_actor", "5.0", "5.0", "YES"],
        ["grad_clip_encoder", "1.0", "1.0", "YES"],
        ["replay_capacity", "800", "800", "YES"],
        ["warmup_steps", "52", "52", "YES"],
        ["warmup_excl_thresh", "128", "128", "YES"],
        ["init_alpha", "0.1", "0.1", "YES"],
        ["alpha_min / max", "1e-4 / 1.0", "1e-4 / 1.0", "YES"],
        ["entropy_scale_factor", "0.7", "0.7", "YES"],
        ["K_max", "102 (Bible)", "110 (config)", "DEVIATED"],
        ["L (lookback)", "60", "60", "YES"],
        ["F (features)", "26 (Bible)", "25 (config)", "DEVIATED"],
        ["per_name_cap", "0.15", "0.15", "YES"],
        ["sector_cap", "0.35", "0.35", "YES"],
        ["lambda_slow", "0.75", "0.75", "YES"],
        ["lambda_tail", "0.4", "0.4", "YES"],
        ["lambda_cost", "1.0", "1.0", "YES"],
        ["lambda_cv", "1.0", "1.0", "YES"],
        ["critic_lr", "3e-4", "3e-4", "YES"],
        ["actor_lr", "1e-4", "1e-4", "YES"],
        ["encoder_lr", "3e-4", "3e-4", "YES"],
        ["encoder_wd", "1e-4", "1e-4", "YES"],
        ["n_quantiles", "32", "32", "YES"],
        ["attn_num_heads", "4", "4", "YES"],
        ["attn_d_model", "128", "128", "YES"],
        ["tcn_levels", "5", "5", "YES"],
        ["tcn_kernel_size", "3", "3", "YES"],
        ["tcn_channels", "128", "128", "YES"],
        ["ticker_emb_dim", "32", "32", "YES"],
        ["sector_emb_dim", "8", "8", "YES"],
        ["recency_half_life", "3 yrs (156w)", "3 yrs", "YES"],
        ["embargo_weeks", "4", "4", "YES"],
        ["bootstrap_resamples", "10,000", "10,000", "YES"],
        ["norm_clip_threshold", "4", "4.0", "YES"],
        ["norm_window_weeks", "52", "52", "YES"],
        ["n_episodes_per_fold", "3", "3", "YES"],
        ["log_update_cadence", "250", "250", "YES"],
    ]
    story.append(status_table(hp_rows, [1.5*inch, 1.3*inch, 1.3*inch, 0.9*inch]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Deviations Noted:</b>", S["BodyJ"]))
    story.append(Paragraph(
        "<b>K_max = 110 vs Bible 102:</b> Config uses 110 for extra headroom. This is acceptable "
        "and the config's own note explains: 'Set to 110 to safely cover historical NDX peaks of "
        "103-104 members with headroom.' No action needed.",
        S["ApexBullet"], bulletText="\u2022",
    ))
    story.append(Paragraph(
        "<b>F = 25 vs Bible 26:</b> The Bible §0.2 table says F=26, but the feature list in the "
        "Bible's own §3.1.1 table lists only 17 per-asset features (not 18). The config correctly "
        "reflects 17 TS + 8 CS = 25. The Bible number 26 likely included adj_close as a separate "
        "feature which was later subsumed into 'close'. The real issue is features/__init__.py "
        "which declares F=26 and lists adj_close — this file is STALE and must be updated.",
        S["ApexBullet"], bulletText="\u2022",
    ))
    story.append(PageBreak())

    # ===== 4. FEATURE ENGINEERING =====
    story.append(Paragraph("4. Feature Engineering Gap Analysis", S["SectionH2"]))
    story.append(hr())

    story.append(Paragraph("<b>4.1 Per-Asset Time-Series Features (§3.1)</b>", S["SectionH3"]))
    story.append(Paragraph(
        "All 17 per-asset TS features are correctly implemented in per_asset_features.py with "
        "causal rolling windows. Feature computation is consistent with Bible definitions.",
        S["BodyJ"],
    ))
    story.append(Paragraph(
        "<b>Issue — vol_52w min_periods:</b> vol_52w uses min_periods=DAYS_4W (20) instead of "
        "DAYS_52W (260). This means vol_52w will be computed from only 20 data points during the "
        "first ~10 months of each security's history, potentially producing noisy estimates. "
        "All other vol windows use matching min_periods.",
        S["ApexBullet"], bulletText="\u2022",
    ))

    story.append(Paragraph("<b>4.2 Cross-Sectional Features (§3.2)</b>", S["SectionH3"]))
    story.append(Paragraph(
        "All 8 CS features are implemented. The ret_rank_4w feature applies z-score AFTER ranking, "
        "which adds a redundant normalization step (ranks are already in [0,1]). This doesn't cause "
        "errors but is unnecessary.",
        S["BodyJ"],
    ))

    story.append(Paragraph("<b>4.3 Portfolio-State Features — CRITICAL GAP</b>", S["SectionH3"]))
    story.append(Paragraph(
        "All 8 portfolio-state features are <b>STUB ZEROS</b>. The file portfolio_state_features.py "
        "contains only compute_portfolio_state_stub() which returns np.zeros. These features require "
        "the environment loop to compute and are essential for the agent to understand its own "
        "portfolio state. Without these, the agent is blind to turnover costs, drawdown risk, "
        "and portfolio concentration. This is the single most important gap in the codebase.",
        S["BodyJ"],
    ))

    story.append(PageBreak())

    # ===== 5. ENVIRONMENT & REWARD =====
    story.append(Paragraph("5. Environment &amp; Reward Function", S["SectionH1"]))
    story.append(hr())

    story.append(Paragraph("<b>5.1 Trading Environment (§5)</b>", S["SectionH3"]))
    story.append(Paragraph(
        "The TradingEnvironment in trading_env.py implements the weekly rebalance cycle, "
        "4-term cost model, NAV evolution, QQQ tracking, and forced liquidation. "
        "Overall compliance with the Bible is strong.",
        S["BodyJ"],
    ))

    env_issues = [
        ("<b>Portfolio return uses PREVIOUS weights (line 263):</b> r_port_t = dot(self._w_exec, "
         "ret_asset) where self._w_exec is the weight from the PREVIOUS step, not the newly "
         "computed w_exec. Bible §5.3 says r_port_t = sum(w_exec_t_i * return_i). However, this "
         "is a common design choice in weekly rebalance models — the new weights only take effect "
         "at the open of d_{t+1}, so using previous weights for the period d_t→d_{t+1} is correct "
         "if we interpret the return as being earned BEFORE the new rebalance. VERIFY the intended "
         "semantics carefully."),
        ("<b>Forced liquidation redistribution:</b> When assets exit NDX, their weight is "
         "redistributed uniformly to ALL remaining active assets. The Bible says redistribution "
         "follows 'the projection result (w_exec) for that step.' This means the freed NAV "
         "should be redistributed according to the constrained target weights, not uniformly."),
        ("<b>Missing data tracking for §5.7:</b> The missing_L[i] and streak_missing[i] counters "
         "from §5.7 are defined in config (missingness section) but are not implemented in "
         "trading_env.py. The environment only checks mask transitions for forced liquidation, "
         "not explicit missing-data thresholds."),
    ]
    for e in env_issues:
        story.append(Paragraph(e, S["ApexBullet"], bulletText="\u2022"))

    story.append(Paragraph("<b>5.2 Reward Function (§6)</b>", S["SectionH3"]))
    story.append(Paragraph(
        "The 5-term reward function in reward_fn.py is correctly implemented with cold-start "
        "blending per §6.4. Reward clipping is ±5 per §6.5. The double-cost mechanism (costs "
        "reducing r_port_net in Term 1 AND explicit λ_cost penalty in Term 4) is correctly "
        "implemented per the Bible's intentional design.",
        S["BodyJ"],
    ))

    story.append(PageBreak())

    # ===== 6. MODEL ARCHITECTURE =====
    story.append(Paragraph("6. Model Architecture Compliance", S["SectionH1"]))
    story.append(hr())

    story.append(Paragraph("<b>6.1 Shared vs Separate Components (§7.2)</b>", S["SectionH3"]))
    story.append(Paragraph(
        "Bible §7.2 Table clearly states: TCNEncoder = <b>Shared</b>, CrossAssetAttention = "
        "<b>Shared</b>. The implementation uses a shared TCN (correct) but instantiates THREE "
        "independent CrossAssetAttention stacks (actor_attn, q1_attn, q2_attn) with separate "
        "parameters. This is a <b>significant architectural divergence</b> that:",
        S["BodyJ"],
    ))
    arch_impacts = [
        "Approximately <b>triples the attention parameters</b> (~3x more weights to train)",
        "Removes the shared representation learning benefit between actor and critic",
        "May increase overfitting risk given the small replay buffer (800 transitions)",
        "Changes the gradient flow dynamics — critic gradients no longer influence attention",
    ]
    for a in arch_impacts:
        story.append(Paragraph(a, S["ApexBullet"], bulletText="\u2022"))
    story.append(Paragraph(
        "<b>Recommendation:</b> Revert to a single shared CrossAssetAttention stack as specified "
        "in the Bible. This reduces parameters and aligns with the intended design.",
        S["BodyJ"],
    ))

    story.append(Paragraph("<b>6.2 Actor Noise (§4.4)</b>", S["SectionH3"]))
    story.append(Paragraph(
        "Bible §4.4 specifies: 'actor learns exploration scale directly via a log-standard-deviation "
        "head' with 'per-asset log_std' output. The implementation uses a single shared scalar "
        "log_sigma (nn.Parameter) instead of per-asset log_std. This means all assets share the same "
        "exploration noise scale. While this simplifies training, it prevents the agent from learning "
        "asset-specific exploration strategies. Also, Bible says log_std range [-5, 1] with init "
        "-1.5 (σ≈0.22), while implementation uses range [-3.0, 0.5] with init -1.0.",
        S["BodyJ"],
    ))

    story.append(Paragraph("<b>6.3 Critic Input (§7.8)</b>", S["SectionH3"]))
    story.append(Paragraph(
        "Bible §7.8 says critic input is 'concatenation of asset_repr (pooled masked mean over "
        "assets), g_t, and action a_pre.' The implementation feeds concat(state_summary, w_pre) "
        "where state_summary is the dual-pooled attention output. The global context g_t is NOT "
        "explicitly concatenated into the critic input (though it was injected earlier into tcn_out "
        "via global_proj). This is a minor deviation — g_t information IS present but indirectly.",
        S["BodyJ"],
    ))
    story.append(PageBreak())

    # ===== 7. TRAINING PIPELINE =====
    story.append(Paragraph("7. Training Pipeline (SAC / Replay / Warmup)", S["SectionH1"]))
    story.append(hr())

    story.append(Paragraph(
        "The SAC training pipeline in sac_trainer.py and replay_buffer.py is well-implemented "
        "and closely follows Bible §8. Key compliance checks:",
        S["BodyJ"],
    ))

    train_items = [
        ("<b>QR-Huber Loss (§8.1.1):</b> Correctly implemented with pairwise quantile "
         "differences and asymmetric Huber weighting. MATCH."),
        ("<b>Actor Loss (§8.1.2):</b> Uses reparameterization in logit space with min(Q1,Q2) "
         "aggregation. Pre-projection entropy. MATCH."),
        ("<b>Alpha Loss (§8.1.3):</b> Auto-tuning with K-dependent H_target = -ln(K_active) × 0.7. "
         "Log-alpha clamped to [log(1e-4), log(1.0)]. MATCH."),
        ("<b>Polyak Update (§8.10):</b> Applied after EVERY critic step (not just policy_delay). "
         "τ = 0.005. MATCH."),
        ("<b>Gradient Clipping (§8.11):</b> critic=1.0, encoder=1.0, actor=5.0. MATCH."),
        ("<b>Optimizer Config (§8.9):</b> Separate Adam/AdamW per component. Embedding params "
         "excluded from weight decay. MATCH."),
        ("<b>Replay Buffer (§8.3):</b> Circular buffer, exponential recency weighting with "
         "calendar-date age (not insertion order). Warmup exclusion after 128 policy transitions. "
         "n-step boundary drop for warmup. All MATCH."),
        ("<b>Transition Augmentation (§8.4):</b> Reward noise (0.01 × buffer_std) applied to "
         "critic path only. Observation augmentation factor stored but actual per-feature "
         "noise application is delegated to obs reconstruction. MATCH."),
    ]
    for item in train_items:
        story.append(Paragraph(item, S["ApexBullet"], bulletText="\u2022"))

    story.append(Paragraph("<b>Training Issues Found:</b>", S["SectionH3"]))
    train_issues = [
        ("<b>Obs augmentation uses fixed factor, not per-feature std:</b> Bible §8.4.2 says "
         "ε_obs[f] ~ N(0, (0.015 × per_feature_std[f])²). The implementation in _reconstruct_obs "
         "uses a flat aug_noise_factor for all features, not per-feature standard deviations. "
         "The per_feature_std is stored in the buffer but never used during reconstruction."),
        ("<b>Encoder optimizer updated twice per actor+critic step:</b> Both _critic_step and "
         "_actor_alpha_step call enc_optimizer.zero_grad() → backward → step. This means "
         "encoder parameters receive TWO gradient steps per update when both critic and actor "
         "fire. This is architecturally questionable — the encoder is trained by both signals "
         "but the zero_grad in _actor_alpha_step wipes any leftover encoder gradients from the "
         "critic step. Verify this is the intended behavior."),
    ]
    for item in train_issues:
        story.append(Paragraph(item, S["ApexBullet"], bulletText="\u2022"))
    story.append(PageBreak())

    # ===== 8. EVALUATION =====
    story.append(Paragraph("8. Evaluation Framework", S["SectionH1"]))
    story.append(hr())
    story.append(Paragraph(
        "The evaluation package (fold_manager, metrics, bootstrap, baselines, leakage_suite, "
        "checkpoint_selector) is comprehensive and well-aligned with Bible §9.",
        S["BodyJ"],
    ))

    eval_items = [
        ("<b>Fold 1 train_start:</b> Bible Table says 2004-01-01, both config and fold_manager "
         "say 2005-01-01. This loses one year of training data for fold 1."),
        ("<b>Fold 8 test_end:</b> Config says 'present', fold_manager says None. Both correctly "
         "handle open-ended evaluation."),
        ("<b>All primary/secondary/tertiary metrics:</b> Implemented in metrics.py. Excess CAGR, "
         "Sortino, Max DD, Sharpe, Turnover, CVaR, Hit Rate, Rank IC all present."),
        ("<b>Block bootstrap:</b> Moving-block with auto block_length = floor(T^(1/3)). 10K "
         "resamples. Correct implementation."),
        ("<b>Leakage suite:</b> All 5 trap tests (temporal, normalizer, membership, embargo, "
         "n-step boundary) implemented."),
    ]
    for item in eval_items:
        story.append(Paragraph(item, S["ApexBullet"], bulletText="\u2022"))

    # ===== 9. INFERENCE =====
    story.append(Paragraph("9. Inference &amp; Paper Trading", S["SectionH1"]))
    story.append(hr())
    story.append(Paragraph(
        "The inference package is comprehensive: checkpoint_loader, guardrails, "
        "missing_data_handler, alert_system, paper_trade_loop, live_data_adapter.",
        S["BodyJ"],
    ))
    story.append(Paragraph(
        "All Bible §12 guardrails are implemented: feasibility check, mask integrity, universe "
        "validity, stale data guard, NAV plausibility. Alert severity levels match §10.6.",
        S["BodyJ"],
    ))

    # ===== 10. LOGGING =====
    story.append(Paragraph("10. Logging &amp; Observability", S["SectionH1"]))
    story.append(hr())
    story.append(Paragraph(
        "ApexLogger implements all 7 metric categories at the correct cadences (every 250 updates, "
        "every env step, per fold, cross-fold). All 7 regression alarms from §10.6 Table 51 "
        "are defined with correct severity levels. Required plots list is maintained.",
        S["BodyJ"],
    ))
    story.append(PageBreak())

    # ===== 11. CODE BUGS & ISSUES =====
    story.append(Paragraph("11. Code Bugs &amp; Issues Found", S["SectionH1"]))
    story.append(hr())

    bugs = [
        (severity_tag("CRITICAL"),
         "<b>features/__init__.py F_TOTAL=26 mismatch:</b> "
         "TS_FEATURE_NAMES in __init__.py includes 'adj_close' (18 features), giving F_TOTAL=26. "
         "But feature_panel.py and per_asset_features.py produce only 17 TS features (no adj_close), "
         "giving F_TOTAL=25. Config says F=25. If any code imports F_TOTAL from __init__.py, it will "
         "create tensor dimension mismatches. <b>Fix:</b> Remove 'adj_close' from __init__.py "
         "TS_FEATURE_NAMES and set F_TOTAL=25."),

        (severity_tag("HIGH"),
         "<b>vol_52w min_periods too low:</b> "
         "per_asset_features.py line 122: vol_52w uses min_periods=DAYS_4W (20) instead of "
         "DAYS_52W. This produces noisy annualized volatility estimates from only 20 data points. "
         "<b>Fix:</b> Change to min_periods=DAYS_52W//2 or similar reasonable minimum."),

        (severity_tag("HIGH"),
         "<b>Obs augmentation ignores per-feature std:</b> "
         "sac_trainer.py _reconstruct_obs applies flat noise factor to all features uniformly. "
         "Bible §8.4.2 requires per-feature scaling: ε[f] ~ N(0, (0.015 × std[f])²). "
         "<b>Fix:</b> Use buffer.per_feature_std in the noise generation."),

        (severity_tag("MEDIUM"),
         "<b>Forced liquidation redistribution is uniform:</b> "
         "trading_env.py line 230: freed weight redistributed uniformly to all active assets. "
         "Bible §2.6 says redistribute 'according to the projection result.' "
         "<b>Fix:</b> After zeroing forced-liq slots, re-run the ConstraintProjector."),

        (severity_tag("MEDIUM"),
         "<b>market_data.py missing mask_panel and x_panel/g_panel passthrough:</b> "
         "build_market_data does not return mask_panel, x_panel, or g_panel in its result dict. "
         "The environment constructor expects these. The synthetic data builder includes them. "
         "<b>Fix:</b> Add these to the return dict from the panel npz."),

        (severity_tag("MEDIUM"),
         "<b>Sector ID lookup is cumulative (not point-in-time):</b> "
         "market_data._build_sector_ids builds sid_to_sector by iterating ALL snapshots and "
         "overwriting. This means the LAST snapshot's sector mapping is used for ALL dates, "
         "not the as-of sector at each date. <b>Fix:</b> Use date-specific sector lookups."),

        (severity_tag("LOW"),
         "<b>Exception silently caught in compute_all_securities_features:</b> "
         "per_asset_features.py line 206: bare 'except Exception: pass' silently drops any "
         "security that raises an error during feature computation. This hides bugs. "
         "<b>Fix:</b> Log warnings or at minimum re-raise unexpected exceptions."),

        (severity_tag("LOW"),
         "<b>config metadata says bible_version v3:</b> "
         "master_config.yaml metadata.bible_version = 'v3' but the actual file is "
         "project_apex_bible_v5.docx. <b>Fix:</b> Update to 'v5'."),
    ]
    for sev, desc in bugs:
        story.append(Paragraph(f"{sev} {desc}", S["ApexBullet"], bulletText="\u2022"))
    story.append(PageBreak())

    # ===== 12. DATA PIPELINE =====
    story.append(Paragraph("12. Data Pipeline Concerns", S["SectionH1"]))
    story.append(hr())

    data_issues = [
        ("<b>Parquet files are empty stubs:</b> daily_bars.parquet (133 bytes), "
         "macro_features.parquet (132 bytes), ndx_membership.parquet (130 bytes), "
         "ticker_alias.parquet (130 bytes), trading_calendar.parquet (130 bytes). These are "
         "placeholder files. The system CANNOT run until real data is loaded."),
        ("<b>CSV data exists but needs conversion:</b> NDX_Membership.csv (209KB) and "
         "Ticker_Sector_AnnualUpdate.csv (390KB) contain real data. The update_parquets.py "
         "script exists to convert these, and a panels_v2/ directory with 17 items suggests "
         "some panel building has occurred."),
        ("<b>ticker_alias table not implemented:</b> Bible §2.3 requires a ticker alias "
         "table mapping historical ticker symbols to permanent security_ids. This file "
         "exists as a stub parquet but no code uses it for reconciliation."),
        ("<b>adj_factor usage:</b> market_data.py correctly notes that close/open in "
         "daily_bars are already adjusted, and adj_factor is for verification only. "
         "This is consistent with Bible §2.1.1."),
    ]
    for d in data_issues:
        story.append(Paragraph(d, S["ApexBullet"], bulletText="\u2022"))
    story.append(PageBreak())

    # ===== 13. RECOMMENDATIONS =====
    story.append(Paragraph("13. Recommendations — Priority Ranked", S["SectionH1"]))
    story.append(hr())

    recs = [
        ("P0 — BLOCKER",
         "Populate real data",
         "Load real daily_bars, macro_features, ndx_membership, and trading_calendar "
         "parquet files. Without data, nothing runs. Use update_parquets.py or equivalent ETL."),
        ("P0 — BLOCKER",
         "Fix F_TOTAL mismatch in __init__.py",
         "Remove adj_close from TS_FEATURE_NAMES in features/__init__.py. Set F_TS=17, "
         "F_TOTAL=25. This prevents dimension errors across the codebase."),
        ("P1 — HIGH",
         "Implement portfolio-state features",
         "Replace compute_portfolio_state_stub with real computation. Wire in environment "
         "state (w_exec, NAV, costs) to produce the 8 portfolio-state features for g_t. "
         "This is critical for agent performance."),
        ("P1 — HIGH",
         "Revert to shared CrossAssetAttention",
         "Replace three separate attention stacks with a single shared stack as specified in "
         "Bible §7.2. This reduces parameters and prevents overfitting on small buffer."),
        ("P1 — HIGH",
         "Fix per-feature obs augmentation",
         "Use buffer.per_feature_std in _reconstruct_obs to scale noise per feature as "
         "specified in Bible §8.4.2."),
        ("P1 — HIGH",
         "Fix vol_52w min_periods",
         "Change min_periods from DAYS_4W to a sensible fraction of DAYS_52W (e.g., "
         "DAYS_52W // 2 = 130)."),
        ("P2 — MEDIUM",
         "Consider per-asset log_std in actor",
         "Bible specifies per-asset log_std head with range [-5, 1] and init -1.5. "
         "Current shared scalar simplifies training but limits exploration diversity."),
        ("P2 — MEDIUM",
         "Fix forced liquidation redistribution",
         "Re-run ConstraintProjector after zeroing forced-liq slots instead of uniform "
         "redistribution."),
        ("P2 — MEDIUM",
         "Fix sector ID point-in-time lookup",
         "Use date-specific sector code mapping in market_data._build_sector_ids instead "
         "of always using the latest snapshot."),
        ("P2 — MEDIUM",
         "Correct Fold 1 train_start",
         "Verify whether 2004-01-01 (Bible) or 2005-01-01 (config) is correct. If data "
         "starts in 2004, use 2004-01-01."),
        ("P3 — LOW",
         "Update bible_version in config metadata",
         "Change metadata.bible_version from 'v3' to 'v5'."),
        ("P3 — LOW",
         "Add logging for silenced exceptions",
         "Replace bare except:pass in compute_all_securities_features with proper logging."),
        ("P3 — LOW",
         "Verify encoder double-update semantics",
         "Confirm the design intent of encoder receiving separate gradient steps from critic "
         "and actor paths. Document if intentional."),
    ]

    rec_rows = [["Priority", "Action", "Details"]]
    for pri, action, details in recs:
        rec_rows.append([pri, action, details])

    story.append(status_table(rec_rows, [1.0*inch, 1.6*inch, 4.2*inch]))
    story.append(PageBreak())

    # ===== 14. PRE-RUN CHECKLIST =====
    story.append(Paragraph("14. Pre-Run Checklist", S["SectionH1"]))
    story.append(hr())
    story.append(Paragraph(
        "Before executing the first training fold, ensure all items below are addressed:",
        S["BodyJ"],
    ))

    checklist = [
        "[ ] Real parquet data files populated (daily_bars, macro_features, ndx_membership, trading_calendar)",
        "[ ] F_TOTAL mismatch in features/__init__.py fixed (set to 25)",
        "[ ] Portfolio-state features implemented (or confirmed acceptable to run with zeros for first pass)",
        "[ ] Shared attention architecture decision made and implemented",
        "[ ] vol_52w min_periods corrected",
        "[ ] Per-feature observation augmentation implemented",
        "[ ] End-to-end integration test (e2e_runner.py) passes with no exceptions",
        "[ ] Leakage suite passes on fold 1",
        "[ ] Unit test suite (all test_phase*_gate*.py) passes",
        "[ ] Deterministic seed verification: two identical runs produce identical results",
        "[ ] GPU availability confirmed if planning full 8-fold run (single-GPU per config)",
        "[ ] Logging output directory configured and writable",
        "[ ] Config validation passes: ProjectConfig.from_yaml('config/master_config.yaml')",
    ]
    for c in checklist:
        story.append(Paragraph(c, S["ApexBullet"], bulletText=""))

    story.append(Spacer(1, 0.5*inch))
    story.append(hr())
    story.append(Paragraph(
        "End of Report. Generated automatically by Project Apex audit tooling.",
        S["SmallNote"],
    ))

    # Build PDF
    doc.build(story)
    print(f"\nReport generated: {output_path}")


if __name__ == "__main__":
    build_report("reports/project_apex_audit_report.pdf")
