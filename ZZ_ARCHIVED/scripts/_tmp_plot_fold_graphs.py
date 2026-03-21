from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path('/Users/samscola/Desktop/project-apex')
RESULTS = ROOT / 'results'
OUT = RESULTS / 'graphs'
OUT.mkdir(parents=True, exist_ok=True)

# Plot 1: Per-fold OOS cumulative returns
fig, axes = plt.subplots(4, 2, figsize=(14, 14))
axes = axes.flatten()
for fold in range(1, 9):
    df = pd.read_csv(RESULTS / f'fold_{fold}' / 'oos_returns.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    port_cum = (1.0 + df['portfolio_return']).cumprod()
    qqq_cum = (1.0 + df['qqq_return']).cumprod()

    ax = axes[fold - 1]
    ax.plot(df['date'], port_cum, label='Portfolio', linewidth=1.8)
    ax.plot(df['date'], qqq_cum, label='QQQ', linewidth=1.5, alpha=0.85)
    ax.set_title(f'Fold {fold}')
    ax.grid(True, alpha=0.25)

axes[0].legend(loc='best')
fig.suptitle('OOS Cumulative Return by Fold (Portfolio vs QQQ)', fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / 'oos_cumulative_returns_8folds.png', dpi=180)
plt.close(fig)

# Plot 2: Average OOS cumulative return over normalized week index
frames = []
for fold in range(1, 9):
    df = pd.read_csv(RESULTS / f'fold_{fold}' / 'oos_returns.csv').reset_index(drop=True)
    df['t'] = range(len(df))
    frames.append(df[['t', 'portfolio_return', 'qqq_return']])
all_df = pd.concat(frames, ignore_index=True)

avg = all_df.groupby('t', as_index=False)[['portfolio_return', 'qqq_return']].mean()
avg_port = (1.0 + avg['portfolio_return']).cumprod()
avg_qqq = (1.0 + avg['qqq_return']).cumprod()

plt.figure(figsize=(9, 5))
plt.plot(avg['t'], avg_port, label='Avg Portfolio', linewidth=2)
plt.plot(avg['t'], avg_qqq, label='Avg QQQ', linewidth=2)
plt.title('Average OOS Cumulative Return Across 8 Folds')
plt.xlabel('Week index within fold')
plt.ylabel('Cumulative growth')
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / 'oos_average_cumulative_return.png', dpi=180)
plt.close()

# Plot 3: Training diagnostics by fold
fig, axes = plt.subplots(4, 2, figsize=(14, 14))
axes = axes.flatten()
for fold in range(1, 9):
    tr = pd.read_csv(RESULTS / f'fold_{fold}' / 'training_log.csv').sort_values('update_num')
    td = tr['td_error'].rolling(100, min_periods=1).median()
    q1 = tr['q1_loss'].rolling(100, min_periods=1).median()
    q2 = tr['q2_loss'].rolling(100, min_periods=1).median()

    ax = axes[fold - 1]
    ax.plot(tr['update_num'], td, label='td_error (roll med 100)', linewidth=1.6)
    ax.plot(tr['update_num'], q1, label='q1_loss (roll med 100)', linewidth=1.2, alpha=0.85)
    ax.plot(tr['update_num'], q2, label='q2_loss (roll med 100)', linewidth=1.2, alpha=0.85)
    ax.set_title(f'Fold {fold}')
    ax.grid(True, alpha=0.25)

axes[0].legend(loc='best', fontsize=8)
fig.suptitle('Training Diagnostics by Fold', fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / 'training_diagnostics_8folds.png', dpi=180)
plt.close(fig)

print('Generated graphs:')
for p in sorted(OUT.glob('*.png')):
    print(p)
