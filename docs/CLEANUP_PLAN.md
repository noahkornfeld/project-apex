# Project Apex - Cleanup Plan

## Current State Analysis

Your project has accumulated temporary files, diagnostic scripts, and multiple result sets from different training runs. Here's a structured cleanup plan.

---

## 📁 FILES TO DELETE (Safe to Remove)

### Temporary Diagnostic Scripts (in `/scripts/`)
These were created for debugging specific issues and are no longer needed:
- `_diag2.py` - Temporary diagnostic
- `_diag_extreme_returns.py` - Extreme returns analysis
- `_final_results.py` - Ad-hoc results script
- `_show_corrected_results.py` - One-time correction script
- `_smoke_test_fixes.py` - Temporary test
- `_tmp_plot_fold_graphs.py` - Temporary plotting
- `_verify_fixes.py` - One-time verification

**Action:** Delete all `scripts/_*.py` files (keep `build_all_panels.py`)

### Root-Level Analysis Scripts
- `analyze_vs_qqq.py` - One-time analysis (results documented)
- `recalculate_metrics.py` - One-time metrics fix (already run)
- `check_step_1250.py` - Diagnostic for old bug (if exists)

**Action:** Move to `scripts/archive/` or delete

### Log Files
- `run_log.txt` - Old run logs (52 KB)
- `run_output.log` - Old output logs (90 KB)

**Action:** Archive or delete if no longer needed

### Empty Directories
- `checkpoints_20up/` - Empty directory from old run naming scheme
- `logs/` - May be empty

**Action:** Delete if empty

---

## 📊 RESULTS ORGANIZATION

### Current Structure (Messy)
```
results/
├── run_3/ (27 items)
├── run_4/ (27 items)
```

### Recommended Structure
```
results/
├── baseline_5updates/          # Original run with updates_per_step=5
│   ├── fold_1/
│   ├── fold_2/
│   ├── ...
│   └── cross_fold/
├── improved_20updates/         # Run with updates_per_step=20
│   ├── fold_1/
│   ├── fold_2/
│   ├── ...
│   └── cross_fold/
├── run_3/                      # Your recent run 3
│   └── ...
└── run_4/                      # Your recent run 4
    └── ...
```

**Action:** Rename `run_3` and `run_4` to descriptive names based on what changes they represent

---

## 🗂️ RECOMMENDED FOLDER STRUCTURE

### Create Archive Directories
```
scripts/
├── archive/              # OLD: Move all _*.py files here
│   ├── _diag2.py
│   ├── _diag_extreme_returns.py
│   └── ...
└── build_all_panels.py   # KEEP: Active script

analysis/                 # NEW: Move one-time analysis scripts
├── analyze_vs_qqq.py
├── recalculate_metrics.py
└── ...

docs/                     # NEW: Organize documentation
├── APEX_AUDIT_REPORT.pdf
├── PHASE1_COMPLETE.md
├── PHASE2_VALIDATION_REPORT.md
├── RECOMMENDATIONS_TO_IMPROVE_SHARPE.md
└── bible_content.txt
```

---

## 🧹 CLEANUP COMMANDS

### Step 1: Create Archive Directories
```powershell
New-Item -ItemType Directory -Path "scripts\archive" -Force
New-Item -ItemType Directory -Path "analysis" -Force
New-Item -ItemType Directory -Path "docs" -Force
```

### Step 2: Move Temporary Scripts
```powershell
Move-Item "scripts\_*.py" "scripts\archive\"
```

### Step 3: Move Analysis Scripts
```powershell
Move-Item "analyze_vs_qqq.py" "analysis\"
Move-Item "recalculate_metrics.py" "analysis\"
```

### Step 4: Move Documentation
```powershell
Move-Item "APEX_AUDIT_REPORT.pdf" "docs\"
Move-Item "PHASE1_COMPLETE.md" "docs\"
Move-Item "PHASE2_VALIDATION_REPORT.md" "docs\"
Move-Item "RECOMMENDATIONS_TO_IMPROVE_SHARPE.md" "docs\"
Move-Item "bible_content.txt" "docs\"
Move-Item "Apex_Stage1_Windsurf_Prompt.md" "docs\"
```

### Step 5: Rename Results Folders (Example)
```powershell
# Rename based on what each run represents
Rename-Item "results\run_3" "results\volatility_targeting_run"
Rename-Item "results\run_4" "results\entropy_15_run"
```

### Step 6: Delete Empty Directories
```powershell
Remove-Item "checkpoints_20up" -Force -ErrorAction SilentlyContinue
```

### Step 7: Clean Logs (Optional)
```powershell
# Archive old logs
New-Item -ItemType Directory -Path "logs\archive" -Force
Move-Item "run_log.txt" "logs\archive\" -ErrorAction SilentlyContinue
Move-Item "run_output.log" "logs\archive\" -ErrorAction SilentlyContinue
```

---

## 📝 UPDATE .gitignore

Add these patterns to `.gitignore`:
```
# Temporary analysis scripts
analysis/
scripts/archive/

# Archived logs
logs/archive/

# Temporary result folders (keep only named runs)
results/run_*/
```

---

## ✅ FINAL CLEAN STRUCTURE

After cleanup, your project should look like:
```
Project_Apex/
├── .git/
├── .gitignore
├── README.md
├── CLAUDE.md
├── SETUP_GUIDE_FOR_GIT.md
├── DATA_QUALITY_SUMMARY.md
├── analysis/              # One-time analysis scripts
├── apex_logging/
├── checkpoints/           # Current training checkpoints
├── config/
├── docs/                  # All documentation
├── environment/
├── evaluation/
├── features/
├── inference/
├── integration/
├── logs/
├── model/
├── reports/
├── results/
│   ├── baseline_5updates/
│   ├── improved_20updates/
│   ├── volatility_targeting_run/
│   └── entropy_15_run/
├── scripts/
│   ├── archive/           # Old diagnostic scripts
│   └── build_all_panels.py
├── tests/
├── training/
├── utils/
├── Ticker_Data/
├── monitor_run.py
├── orchestrate_overnight.py
└── run_full_apex.py
```

---

## 🚀 QUICK CLEANUP (Run All Steps)

Save this as `cleanup.ps1` and run it:
```powershell
# Create directories
New-Item -ItemType Directory -Path "scripts\archive","analysis","docs" -Force

# Move files
Move-Item "scripts\_*.py" "scripts\archive\" -Force
Move-Item "analyze_vs_qqq.py","recalculate_metrics.py" "analysis\" -Force
Move-Item "APEX_AUDIT_REPORT.pdf","PHASE*.md","RECOMMENDATIONS*.md","bible_content.txt","Apex_Stage1*.md" "docs\" -Force

# Clean empty dirs
Remove-Item "checkpoints_20up" -Force -ErrorAction SilentlyContinue

Write-Host "Cleanup complete! Review the changes and commit."
```

---

## 📌 WHAT TO KEEP

**Essential Files:**
- All code in `model/`, `training/`, `environment/`, `evaluation/`
- `run_full_apex.py`, `monitor_run.py`, `orchestrate_overnight.py`
- `config/master_config.yaml`
- Current `checkpoints/` and active `results/` folders
- `README.md`, `CLAUDE.md`, `SETUP_GUIDE_FOR_GIT.md`

**Can Delete:**
- Temporary `_*.py` scripts
- Old log files
- Empty directories
- One-time analysis scripts (after archiving)
