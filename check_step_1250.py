import pandas as pd
import numpy as np

# Load daily bars
df = pd.read_parquet('Ticker_Data/daily_bars.parquet')
print(f'Total rows in parquet: {len(df)}')
print(f'Columns: {df.columns.tolist()}')

# Check if 'date' is a column
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    dates = df['date'].unique()
    dates = pd.DatetimeIndex(dates).sort_values()
    print(f'Unique dates: {len(dates)}')
    
    # Fold 3 training period: 2008-01-01 to 2013-12-31
    fold3_train = dates[(dates >= '2008-01-01') & (dates <= '2013-12-31')]
    print(f'\nFold 3 training dates: {len(fold3_train)}')
    
    if len(fold3_train) > 0:
        print(f'Fold 3 start: {fold3_train[0]}')
        print(f'Fold 3 end: {fold3_train[-1]}')
        
        # Step 1250 (daily steps, not weekly)
        if len(fold3_train) > 1250:
            step_1250_date = fold3_train[1250]
            print(f'\nStep 1250 date: {step_1250_date.strftime("%Y-%m-%d")}')
            
            # Check around this date for missing data
            window_start = max(0, 1250 - 10)
            window_end = min(len(fold3_train), 1250 + 10)
            print(f'\nDates around step 1250:')
            for i in range(window_start, window_end):
                print(f'  Step {i}: {fold3_train[i].strftime("%Y-%m-%d")}')
            
            # Check for data quality issues around step 1250
            print('\n--- Checking data quality around step 1250 ---')
            # Compute adj_close from close and adj_factor
            for offset in [-5, -3, -1, 0, 1, 3, 5]:
                idx = 1250 + offset
                if 0 <= idx < len(fold3_train):
                    date = fold3_train[idx]
                    df_at = df[df['date'] == date].copy()
                    df_at['adj_close'] = df_at['close'] * df_at['adj_factor']
                    valid = (df_at['adj_close'] > 0) & (~df_at['adj_close'].isna())
                    print(f'  {date.strftime("%Y-%m-%d")} (step {idx}): {valid.sum()} valid / {len(df_at)} total tickers')
        else:
            print(f'\nFold 3 only has {len(fold3_train)} days, step 1250 is beyond range')
else:
    print('Date column not found. Showing first few rows:')
    print(df.head())
