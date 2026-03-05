"""
Automated Parquet Update Script
================================
Updates daily_bars.parquet and macro_features.parquet with latest data from Yahoo Finance.

Process:
1. Read last date from existing parquets
2. Get current NDX members from ndx_membership.parquet
3. Fetch new daily price data (last_date+1 to today)
4. Fetch new macro features (last_date+1 to today)
5. Apply transformations (adj_open, adj_close, returns, etc.)
6. Append to parquets and save

Usage:
    python update_parquets.py
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Determine DATA_DIR dynamically (script location)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Try to import yfinance, provide helpful error if missing
try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not found. Install with:")
    print("  python3 -m pip install yfinance")
    sys.exit(1)

print("="*80)
print("AUTOMATED PARQUET UPDATE")
print("="*80)
print(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =========================================================================
# 1. READ EXISTING PARQUETS AND DETERMINE UPDATE RANGE
# =========================================================================
print("\n" + "="*80)
print("1. CHECKING EXISTING DATA")
print("="*80)

# Load daily_bars
daily_bars_path = os.path.join(DATA_DIR, 'daily_bars.parquet')
df_bars = pd.read_parquet(daily_bars_path)
last_bars_date = df_bars['date'].max()
print(f"\ndaily_bars.parquet:")
print(f"  Last date: {last_bars_date.date()}")
print(f"  Rows: {len(df_bars):,}")

# Load macro_features
macro_path = os.path.join(DATA_DIR, 'macro_features.parquet')
df_macro = pd.read_parquet(macro_path)
last_macro_date = df_macro['date'].max()
print(f"\nmacro_features.parquet:")
print(f"  Last date: {last_macro_date.date()}")
print(f"  Rows: {len(df_macro):,}")

# Determine update range
start_date = max(last_bars_date, last_macro_date) + timedelta(days=1)
end_date = datetime.now()

print(f"\nUpdate range:")
print(f"  Start: {start_date.date()}")
print(f"  End:   {end_date.date()}")

if start_date >= end_date:
    print("\n✓ Data is already up to date!")
    sys.exit(0)

# =========================================================================
# 2. GET CURRENT NDX MEMBERS
# =========================================================================
print("\n" + "="*80)
print("2. GETTING CURRENT NDX MEMBERS")
print("="*80)

# Load membership
ndx_path = os.path.join(DATA_DIR, 'ndx_membership.parquet')
df_ndx = pd.read_parquet(ndx_path)

# Get most recent snapshot
latest_snapshot_date = df_ndx['date'].max()
current_members = df_ndx[df_ndx['date'] == latest_snapshot_date]

print(f"\nLatest NDX snapshot: {latest_snapshot_date.date()}")
print(f"Current members: {len(current_members)}")

# Get tickers and security_ids
ticker_to_permno = current_members.set_index('ticker')['security_id'].to_dict()
tickers = sorted([t for t in current_members['ticker'].dropna().unique() if pd.notna(t)])

print(f"Tickers to update: {len(tickers)}")
print(f"Sample: {tickers[:5]}")

# =========================================================================
# 3. FETCH NEW DAILY PRICE DATA
# =========================================================================
print("\n" + "="*80)
print("3. FETCHING NEW DAILY PRICE DATA")
print("="*80)

start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')

print(f"\nFetching from Yahoo Finance: {start_str} to {end_str}")

new_bars_list = []
success_count = 0
fail_count = 0

for ticker in tickers:
    try:
        # Get PERMNO
        permno = ticker_to_permno.get(ticker)
        if pd.isna(permno):
            print(f"  {ticker}: No PERMNO, skipping")
            fail_count += 1
            continue
        
        # Fetch data
        yf_ticker = yf.Ticker(ticker)
        hist = yf_ticker.history(start=start_str, end=end_str, auto_adjust=False)
        
        if len(hist) == 0:
            print(f"  {ticker}: No data")
            fail_count += 1
            continue
        
        # Process
        hist = hist.reset_index()
        hist['date'] = pd.to_datetime(hist['Date'], utc=True).dt.tz_localize(None).dt.normalize()
        
        # Calculate adjustment factor (use Adj Close / Close ratio)
        hist['adj_factor'] = hist['Adj Close'] / hist['Close']
        
        # Calculate adjusted prices
        hist['adj_close'] = hist['Adj Close']
        hist['adj_open'] = hist['Open'] * hist['adj_factor']
        
        # Build rows
        for _, row in hist.iterrows():
            new_bars_list.append({
                'date': row['date'],
                'security_id': int(permno),
                'ticker': ticker,
                'open': row['adj_open'],
                'close': row['adj_close'],
                'volume': int(row['Volume']) if pd.notna(row['Volume']) else None,
                'adj_factor': row['adj_factor'],
            })
        
        success_count += 1
        if success_count % 20 == 0:
            print(f"  Progress: {success_count}/{len(tickers)} tickers")
        
    except Exception as e:
        print(f"  {ticker}: Error - {str(e)[:50]}")
        fail_count += 1

print(f"\n✓ Fetched {success_count} tickers, {fail_count} failed")

if len(new_bars_list) == 0:
    print("⚠ No new price data fetched")
else:
    df_new_bars = pd.DataFrame(new_bars_list)
    print(f"New price rows: {len(df_new_bars):,}")
    print(f"Date range: {df_new_bars['date'].min().date()} to {df_new_bars['date'].max().date()}")

# =========================================================================
# 4. FETCH NEW MACRO FEATURES
# =========================================================================
print("\n" + "="*80)
print("4. FETCHING NEW MACRO FEATURES")
print("="*80)

instruments = {
    'QQQ':      {'symbol': 'QQQ',      'col_prefix': 'QQQ'},
    'VIX':      {'symbol': '^VIX',     'col_prefix': 'VIX'},
    '10Y':      {'symbol': '^TNX',     'col_prefix': '10Y_Yield'},
    '3M':       {'symbol': '^IRX',     'col_prefix': '3M_Yield'},
    'Oil':      {'symbol': 'CL=F',     'col_prefix': 'Oil'},
    'Gold':     {'symbol': 'GC=F',     'col_prefix': 'Gold'},
    'Dollar':   {'symbol': 'DX-Y.NYB', 'col_prefix': 'Dollar_Index'},
    'HYG':      {'symbol': 'HYG',      'col_prefix': 'HYG'},
}

macro_frames = {}

for name, info in instruments.items():
    symbol = info['symbol']
    prefix = info['col_prefix']
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_str, end=end_str, auto_adjust=False)
        
        if len(df) == 0:
            print(f"  {name} ({symbol}): No data")
            continue
        
        df = df.reset_index()
        df['date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None).dt.normalize()
        
        df_out = pd.DataFrame({'date': df['date']})
        df_out[f'{prefix}_Close'] = df['Close'].values
        
        if name in ['QQQ', 'HYG', 'Oil', 'Gold', 'Dollar']:
            df_out[f'{prefix}_Volume'] = df['Volume'].values
        
        df_out = df_out.groupby('date').last().reset_index()
        macro_frames[name] = df_out
        print(f"  {name} ({symbol}): {len(df_out)} days")
        
    except Exception as e:
        print(f"  {name} ({symbol}): Error - {str(e)[:50]}")

# Merge macro frames
if len(macro_frames) == 0:
    print("⚠ No new macro data fetched")
    df_new_macro = None
else:
    df_new_macro = None
    for name, df_inst in macro_frames.items():
        if df_new_macro is None:
            df_new_macro = df_inst
        else:
            df_new_macro = df_new_macro.merge(df_inst, on='date', how='outer')
    
    df_new_macro = df_new_macro.sort_values('date').reset_index(drop=True)
    
    # Calculate derived features
    if '10Y_Yield_Close' in df_new_macro.columns and '3M_Yield_Close' in df_new_macro.columns:
        df_new_macro['Yield_Spread'] = df_new_macro['10Y_Yield_Close'] - df_new_macro['3M_Yield_Close']
    
    if 'VIX_Close' in df_new_macro.columns:
        df_new_macro['VIX_change'] = df_new_macro['VIX_Close'].diff()
        df_new_macro['VIX_pct_change'] = df_new_macro['VIX_Close'].pct_change()
    
    for prefix in ['QQQ', 'Oil', 'Gold', 'Dollar_Index', 'HYG']:
        col = f'{prefix}_Close'
        if col in df_new_macro.columns:
            df_new_macro[f'{prefix}_return'] = df_new_macro[col].pct_change()
            df_new_macro[f'{prefix}_log_return'] = np.log(df_new_macro[col] / df_new_macro[col].shift(1))
    
    # Forward fill
    df_new_macro = df_new_macro.ffill()
    
    print(f"\n✓ New macro rows: {len(df_new_macro):,}")
    print(f"Date range: {df_new_macro['date'].min().date()} to {df_new_macro['date'].max().date()}")

# =========================================================================
# 5. APPEND TO DAILY_BARS AND RECALCULATE LOG_RETURN
# =========================================================================
print("\n" + "="*80)
print("5. APPENDING TO daily_bars.parquet")
print("="*80)

if len(new_bars_list) > 0:
    # Align columns before concatenating
    existing_cols = df_bars.columns.tolist()
    new_cols = df_new_bars.columns.tolist()
    
    # Add missing columns to new data (fill with None)
    for col in existing_cols:
        if col not in new_cols:
            df_new_bars[col] = None
    
    # Reorder new data columns to match existing
    df_new_bars = df_new_bars[existing_cols]
    
    # Convert both to same dtype structure by going through dict
    # This ensures internal pandas array structures match
    df_bars_dict = df_bars.to_dict('records')
    df_new_bars_dict = df_new_bars.to_dict('records')
    
    # Combine and recreate dataframe
    combined_records = df_bars_dict + df_new_bars_dict
    df_bars_combined = pd.DataFrame(combined_records)
    df_bars_combined = df_bars_combined.sort_values(['security_id', 'date']).reset_index(drop=True)
    
    # Remove duplicates (keep last)
    before = len(df_bars_combined)
    df_bars_combined = df_bars_combined.drop_duplicates(subset=['security_id', 'date'], keep='last')
    after = len(df_bars_combined)
    if before != after:
        print(f"Removed {before - after:,} duplicate rows")
    
    print(f"\nCombined daily_bars:")
    print(f"  Rows: {len(df_bars_combined):,} (was {len(df_bars):,}, added {len(df_bars_combined) - len(df_bars):,})")
    print(f"  Date range: {df_bars_combined['date'].min().date()} to {df_bars_combined['date'].max().date()}")
    
    # Save
    df_bars_combined.to_parquet(daily_bars_path, index=False, engine='pyarrow')
    print(f"✓ Saved updated daily_bars.parquet")
else:
    print("No new daily bars to append")

# =========================================================================
# 6. APPEND TO MACRO_FEATURES
# =========================================================================
print("\n" + "="*80)
print("6. APPENDING TO macro_features.parquet")
print("="*80)

if df_new_macro is not None and len(df_new_macro) > 0:
    # Align columns
    existing_cols = df_macro.columns.tolist()
    new_cols = df_new_macro.columns.tolist()
    
    # Add missing columns to new data
    for col in existing_cols:
        if col not in new_cols:
            df_new_macro[col] = None
    
    # Add missing columns to old data (shouldn't happen, but just in case)
    for col in new_cols:
        if col not in existing_cols:
            df_macro[col] = None
    
    # Reorder new data to match existing
    df_new_macro = df_new_macro[existing_cols]
    
    # Convert both to same dtype structure by going through dict
    df_macro_dict = df_macro.to_dict('records')
    df_new_macro_dict = df_new_macro.to_dict('records')
    
    # Combine and recreate dataframe
    combined_records = df_macro_dict + df_new_macro_dict
    df_macro_combined = pd.DataFrame(combined_records)
    df_macro_combined = df_macro_combined.sort_values('date').reset_index(drop=True)
    
    # Remove duplicates
    before = len(df_macro_combined)
    df_macro_combined = df_macro_combined.drop_duplicates(subset=['date'], keep='last')
    after = len(df_macro_combined)
    if before != after:
        print(f"Removed {before - after:,} duplicate rows")
    
    print(f"\nCombined macro_features:")
    print(f"  Rows: {len(df_macro_combined):,} (was {len(df_macro):,}, added {len(df_macro_combined) - len(df_macro):,})")
    print(f"  Date range: {df_macro_combined['date'].min().date()} to {df_macro_combined['date'].max().date()}")
    
    # Save
    df_macro_combined.to_parquet(macro_path, index=False, engine='pyarrow')
    print(f"✓ Saved updated macro_features.parquet")
else:
    print("No new macro features to append")

# =========================================================================
# 7. UPDATE TRADING CALENDAR
# =========================================================================
print("\n" + "="*80)
print("7. UPDATING trading_calendar.parquet")
print("="*80)

if len(new_bars_list) > 0:
    # Reload updated daily_bars to get all dates
    df_bars_updated = pd.read_parquet(daily_bars_path)
    all_dates = sorted(df_bars_updated['date'].unique())
    
    calendar = pd.DataFrame({
        'date': all_dates,
        't_idx': range(len(all_dates)),
    })
    calendar['date'] = pd.to_datetime(calendar['date'])
    calendar['year'] = calendar['date'].dt.isocalendar().year.astype(int)
    calendar['week'] = calendar['date'].dt.isocalendar().week.astype(int)
    calendar['is_week_start'] = ~calendar.duplicated(subset=['year', 'week'], keep='first')
    calendar['week_id'] = calendar.groupby(['year', 'week']).ngroup()
    
    cal_path = f'{DATA_DIR}\\trading_calendar.parquet'
    calendar.to_parquet(cal_path, index=False, engine='pyarrow')
    
    print(f"✓ Updated trading_calendar.parquet")
    print(f"  Trading days: {len(calendar):,}")
    print(f"  Date range: {calendar['date'].min().date()} to {calendar['date'].max().date()}")
else:
    print("No calendar update needed")

# =========================================================================
# FINAL SUMMARY
# =========================================================================
print("\n" + "="*80)
print("UPDATE COMPLETE")
print("="*80)

if len(new_bars_list) > 0 or (df_new_macro is not None and len(df_new_macro) > 0):
    print("\n✓ Parquets updated successfully!")
    print(f"\nUpdated files:")
    if len(new_bars_list) > 0:
        print(f"  - daily_bars.parquet (+{len(new_bars_list):,} rows)")
        print(f"  - trading_calendar.parquet (rebuilt)")
    if df_new_macro is not None and len(df_new_macro) > 0:
        print(f"  - macro_features.parquet (+{len(df_new_macro):,} rows)")
else:
    print("\n✓ No updates needed - data is current!")

print(f"\nRun completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
