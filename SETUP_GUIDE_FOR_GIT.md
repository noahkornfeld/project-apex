# Project Apex - Setup Guide for Coworkers

**A step-by-step guide for non-technical users to get Project Apex running on their Mac.**

---

## 🔄 Day-to-Day Usage (After Initial Setup)

### Running the Data Update Script

**How often:** Weekly (or whenever you need fresh data)

1. **Open Terminal:**
   - Press **Command (⌘) + Space**
   - Type **"Terminal"**
   - Press **Enter**

2. **Navigate to the project:**
   ```bash
   cd ~/Desktop/project-apex/Ticker_Data
   ```

3. **Run the update script:**
   ```bash
   python3 update_parquets.py
   ```

4. **What happens:**
   - Script checks the last date in your data
   - Downloads new data from Yahoo Finance (if available)
   - Updates all parquet files automatically
   - Takes 1-5 minutes depending on how much new data there is

5. **When complete:**
   - You'll see: `✓ Parquets updated successfully!`
   - Your data is now up to date!

---

### Getting Code Updates from GitHub

**When to do this:** When Noah pushes new changes to the repository

1. **Open Terminal**

2. **Navigate to the project:**
   ```bash
   cd ~/Desktop/project-apex
   ```

3. **Download the latest changes:**
   ```bash
   git pull
   ```

4. **You'll see:**
   - A list of files that were updated
   - Or "Already up to date" if nothing changed

---

## 🎯 Quick Reference Commands

### Update Data
```bash
cd ~/Desktop/project-apex/Ticker_Data
python3 update_parquets.py
```

### Get Latest Code Changes
```bash
cd ~/Desktop/project-apex
git pull
```

### Check Git Version
```bash
git --version
```

### Check Python Version
```bash
python3 --version
```

---

## 📋 One-Time Setup (Do This First)

**What You'll Need:**
- A Mac computer (macOS 10.15 or later)
- Internet connection
- About 15 minutes

---

## 🚀 Initial Setup Steps

### Step 1: Install Git

Git is a tool that lets you download and sync code from GitHub.

**Good news:** Git is usually pre-installed on Mac! Let's check:

1. **Open Terminal:**
   - Press **Command (⌘) + Space**
   - Type **"Terminal"**
   - Press **Enter**

2. **Check if Git is installed:**
   ```bash
   git --version
   ```

3. **What you'll see:**
   - If Git is installed: `git version 2.x.x`
   - If not installed: A popup will appear asking to install Command Line Developer Tools
     - Click **"Install"**
     - Wait 5-10 minutes for installation
     - Try `git --version` again

---

### Step 2: Install Python

Python is the programming language that runs the data update scripts.

**Good news:** Python 3 is usually pre-installed on Mac! Let's check:

1. **Check if Python 3 is installed:**
   ```bash
   python3 --version
   ```

2. **What you'll see:**
   - If installed: `Python 3.x.x`
   - If not installed (rare), install via Homebrew:
     ```bash
     # Install Homebrew first (if needed)
     /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
     
     # Then install Python
     brew install python3
     ```

3. **Verify:**
   ```bash
   python3 --version
   ```
   You should see: `Python 3.x.x`

---

### Step 3: Download the Project

1. **Open Terminal** (if not already open)

2. **Navigate to Your Desktop:**
   ```bash
   cd ~/Desktop
   ```

3. **Download the Project:**
   ```bash
   git clone https://github.com/noahkornfeld/project-apex.git
   ```
   
   This will create a folder called `project-apex` on your Desktop.

4. **Go into the Project Folder:**
   ```bash
   cd project-apex
   ```

---

### Step 4: Install Required Python Packages

The project needs some additional Python tools to work.

1. **In Terminal (still in the project-apex folder), type:**
   ```bash
   pip3 install pandas numpy yfinance pyarrow
   ```

2. **Wait for installation to complete** (takes 1-2 minutes)

3. **You should see:** `Successfully installed pandas-x.x.x numpy-x.x.x ...`

---

### Step 5: Verify Everything Works

1. **Check that all files are there:**
   ```bash
   ls
   ```
   
   You should see:
   - `README.md`
   - `DATA_QUALITY_SUMMARY.md`
   - `Ticker_Data` folder
   - And other files

2. **Go into the Ticker_Data folder:**
   ```bash
   cd Ticker_Data
   ```

3. **Check the parquet files:**
   ```bash
   ls *.parquet
   ```
   
   You should see 5 parquet files:
   - `daily_bars.parquet`
   - `ndx_membership.parquet`
   - `macro_features.parquet`
   - `trading_calendar.parquet`
   - `ticker_alias.parquet`

**✅ Setup Complete! You're ready to use the project.**

---

## 🆘 Troubleshooting

### Problem: "command not found: git"
**Solution:**
1. Install Command Line Developer Tools:
   ```bash
   xcode-select --install
   ```
2. Click "Install" in the popup
3. Wait for installation to complete
4. Try `git --version` again

### Problem: "command not found: python3"
**Solution:**
1. Install Homebrew:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. Install Python:
   ```bash
   brew install python3
   ```
3. Try `python3 --version` again

### Problem: "No module named 'pandas'"
**Solution:**
```bash
pip3 install pandas numpy yfinance pyarrow
```

### Problem: "Permission denied" when running git clone
**Solution:**
1. Make sure you're in your home directory:
   ```bash
   cd ~/Desktop
   git clone https://github.com/noahkornfeld/project-apex.git
   ```

### Problem: Update script says "No new data"
**Solution:**
- This is normal! It means your data is already up to date
- The script only downloads data for dates you don't have yet

### Problem: "Failed to fetch" error during update
**Solution:**
- Check your internet connection
- Some tickers might be delisted or unavailable
- The script will continue with other tickers
- This is usually not a problem

---

## 📁 Understanding the Files

### What's in the Project Folder?

```
project-apex/
├── README.md                          # Project overview
├── DATA_QUALITY_SUMMARY.md            # Data quality report
├── SETUP_GUIDE_FOR_COWORKERS.md       # This file!
├── project_apex_bible_v3.docx         # Full project specs
└── Ticker_Data/
    ├── update_parquets.py             # Script to update data
    ├── daily_bars.parquet             # Daily stock prices
    ├── ndx_membership.parquet         # NDX member list
    ├── macro_features.parquet         # Economic indicators
    ├── trading_calendar.parquet       # Trading days calendar
    └── ticker_alias.parquet           # Ticker change history
```

**Location on your Mac:** `~/Desktop/project-apex/`

### What are Parquet Files?

- Parquet files (`.parquet`) are compressed data files
- They're like Excel files but much faster and smaller
- You can't open them in Excel, but Python can read them easily
- They contain all the stock market data for the project

---


---

## 💡 Tips

1. **Run updates weekly** to keep data current
2. **Don't edit the parquet files** - they're generated automatically
3. **If something breaks**, try `git pull` to get the latest fixes
4. **Keep Terminal open** while the update script runs
5. **Internet required** - the script downloads data from Yahoo Finance
6. **Use `python3` not `python`** on Mac (Python 2 vs Python 3)

---

## 📞 Need Help?

If you're stuck:
1. Check the Troubleshooting section above
2. Try closing and reopening Terminal
3. Try restarting your Mac
4. Contact Noah for help

---

## ✅ Setup Checklist

Use this to make sure everything is set up correctly:

- [ ] Git installed (`git --version` works)
- [ ] Python installed (`python3 --version` works)
- [ ] Project downloaded (`cd ~/Desktop/project-apex` works)
- [ ] Python packages installed (`pip3 install pandas numpy yfinance pyarrow`)
- [ ] Can see 5 parquet files in `Ticker_Data` folder
- [ ] Update script runs successfully (`python3 update_parquets.py`)

**If all boxes are checked, you're ready to go!** 🎉

---

*Last Updated: March 4, 2026*
