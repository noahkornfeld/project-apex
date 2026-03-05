# Project Apex: Reinforcement Learning for Portfolio Management

A Soft Actor-Critic (SAC) based reinforcement learning system for dynamic portfolio management of Nasdaq-100 (NDX) index constituents.

## 📋 Overview

Project Apex implements a state-of-the-art RL agent that learns optimal portfolio allocation strategies through interaction with historical market data. The system uses point-in-time data to prevent look-ahead bias and incorporates macro regime indicators for robust decision-making.

### Key Features
- **22+ years of historical data** (2004-2026)
- **Point-in-time NDX membership** tracking with sector constraints
- **Macro regime indicators** (VIX, yields, commodities, credit spreads)
- **Total-return adjusted prices** (dividends + splits)
- **Automated daily updates** via Yahoo Finance
- **Production-ready parquet datasets** optimized for ML pipelines

## 🎯 Objectives

1. **Outperform QQQ benchmark** with risk-adjusted returns
2. **Learn dynamic allocation** without hand-crafted rules
3. **Enforce realistic constraints** (sector caps, turnover limits, transaction costs)
4. **Prevent look-ahead leakage** through strict as-of data usage

## 📊 Dataset

### Core Parquet Files

| File | Rows | Description |
|------|------|-------------|
| `daily_bars.parquet` | 1,142,839 | OHLCV data for 316 securities |
| `ndx_membership.parquet` | 4,356 | Point-in-time NDX snapshots (44 periods) |
| `macro_features.parquet` | 5,593 | 9 macro instruments with derived features |
| `trading_calendar.parquet` | 5,577 | Date-to-index mapping for temporal alignment |
| `ticker_alias.parquet` | 381 | Historical ticker changes tracking |

### Data Coverage
- **Time span:** January 2, 2004 - March 3, 2026
- **Securities:** 316 unique PERMNOs, 340 tickers (including historical changes)
- **Snapshots:** 44 semi-annual NDX membership updates
- **Macro instruments:** QQQ, VIX, 10Y/3M Yields, Oil, Gold, Dollar Index, HYG

### Data Quality
- ✅ 100% sector code coverage (11 GICS sectors)
- ✅ 99.98% trading day coverage for macro features
- ✅ All prices total-return adjusted (CFACPR)
- ✅ Point-in-time membership (as-of rule enforced)
- ✅ Comprehensive data validation and cleaning

## 🏗️ Architecture

### Observation Space
- **Per-asset features** `x_t`: [L, K_max, F] tensor
  - L = lookback weeks
  - K_max = max portfolio size (102-104)
  - F = features per asset (momentum, volatility, volume, sector)
  
- **Global context** `g_t`: [D_global] vector
  - Macro regime indicators (VIX, yields, commodities)
  - Portfolio state (holdings, cash, turnover)
  - Benchmark tracking (QQQ returns)

### Action Space
- **Continuous allocation weights** for each asset
- Constraints: long-only, sector caps, turnover limits

### Reward Signal
- Risk-adjusted returns (Sharpe ratio)
- Transaction cost penalties
- Turnover regularization

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy yfinance pyarrow
```

### Data Updates
Run the automated update script to fetch latest data:
```bash
cd Ticker_Data
python update_parquets.py
```

This will:
- Fetch new daily prices for current NDX members
- Update macro features for all 9 instruments
- Rebuild trading calendar with new dates
- Handle duplicates and missing data automatically

### Project Structure
```
Project_Apex/
├── README.md                          # This file
├── DATA_QUALITY_SUMMARY.md            # Comprehensive data quality report
├── project_apex_bible_v3.docx         # Full project specifications
├── bible_content.txt                  # Bible text version
└── Ticker_Data/
    ├── update_parquets.py             # Automated daily update script
    ├── daily_bars.parquet             # OHLCV data
    ├── ndx_membership.parquet         # Point-in-time membership
    ├── macro_features.parquet         # Macro regime indicators
    ├── trading_calendar.parquet       # Date-to-index mapping
    ├── ticker_alias.parquet           # Ticker change tracking
    ├── NDX_Membership.csv             # Source membership data
    └── Ticker_Sector_AnnualUpdate.csv # GICS sector mapping
```

## 📈 Data Pipeline

### 1. Daily Price Bars (`daily_bars.parquet`)
- **Schema:** date, security_id, ticker, open, close, volume, adj_factor
- **Adjustment:** Total-return (dividends + splits via CFACPR)
- **Coverage:** 316 securities, 5,577 trading days

### 2. NDX Membership (`ndx_membership.parquet`)
- **Schema:** date, security_id, ticker, sector_code, weight
- **Snapshots:** Semi-annual (June 30, December 31)
- **Sectors:** 11 GICS sectors with 100% coverage
- **As-of rule:** Backward fill only (no look-ahead)

### 3. Macro Features (`macro_features.parquet`)
- **Instruments:** QQQ, VIX, 10Y/3M Yields, Oil, Gold, Dollar Index, HYG
- **Derived features:** Yield spread, VIX changes, returns, log returns
- **Coverage:** 99.98% of trading days
- **Special handling:** HYG starts April 2007 (ETF launch)

### 4. Trading Calendar (`trading_calendar.parquet`)
- **Purpose:** Canonical date-to-index mapping
- **Features:** t_idx, year, week, is_week_start, week_id
- **Usage:** Temporal alignment and rebalance week identification

## 🔧 Data Quality

### Validation
- ✅ No ghost entries (null ticker + null prices removed)
- ✅ No duplicates in membership snapshots
- ✅ All sector codes backfilled (manual research + Compustat)
- ✅ Ticker changes tracked (FB→META, SUNW→JAVA, etc.)
- ✅ Extreme returns verified (M&A events, bankruptcies)

### Known Characteristics
- **418 zero-volume rows**: Illiquid securities (expected)
- **71 extreme returns (>50%)**: Corporate events (verified)
- **HYG nulls before 2007**: ETF launch date (expected)
- **K_max exceeded in 3 snapshots**: 104 members vs 102 limit

See `DATA_QUALITY_SUMMARY.md` for comprehensive analysis.

## 📚 Documentation

- **`DATA_QUALITY_SUMMARY.md`**: Comprehensive data quality report
- **`project_apex_bible_v3.docx`**: Full project specifications
- **`bible_content.txt`**: Text version of project Bible

## 🛠️ Maintenance

### Automated Updates
The `update_parquets.py` script handles:
- Daily price data fetching for current NDX members
- Macro feature updates for all 9 instruments
- Trading calendar rebuilding
- Duplicate detection and removal
- Column alignment and forward-filling

### Manual Interventions
Rare cases requiring manual updates:
- NDX membership changes (semi-annual)
- Sector code updates (annual)
- Corporate events (M&A, ticker changes)

## 📊 Next Steps

### Feature Engineering (Bible §3)
- [ ] Per-Asset Time-Series Features (momentum, volatility, volume)
- [ ] Cross-Sectional Features (rank, sector-relative)
- [ ] Macro/Broadcast Features (VIX, yields, commodities)
- [ ] Portfolio-State Features (holdings, cash, turnover)
- [ ] Benchmark Features (QQQ tracking error)

### Dense Panel Construction (Bible §4)
- [ ] Build observation tensor `x_t` [L, K_max, F]
- [ ] Build global context `g_t` [D_global]
- [ ] Implement padding and masking for variable K
- [ ] Create tradability mask (as-of rule enforcement)

### SAC Training (Bible §5)
- [ ] Define reward function (Sharpe ratio + costs)
- [ ] Implement actor-critic networks
- [ ] Set up replay buffer with proper batching
- [ ] Train with transaction cost model
- [ ] Evaluate against QQQ benchmark

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- **Data sources:** CRSP (historical), Yahoo Finance (recent + macro)
- **Sector mapping:** Compustat GICS codes
- **Methodology:** Soft Actor-Critic (SAC) for continuous control

---

**Status:** ✅ Data pipeline complete, ready for feature engineering phase  
**Last Updated:** March 4, 2026  
**Data Quality Score:** 99.5%
