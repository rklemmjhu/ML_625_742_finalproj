import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.api import OLS, add_constant
from statsmodels.tsa.stattools import grangercausalitytests

# ── File: scripts/returns_analysis.py ────────────────────────────────────────
# Computes log-returns, scans lead/lag on returns, runs OLS & Granger tests

# Configuration
datasets_dir = 'datasets/'
paths = {
    'box_office': f'{datasets_dir}Daily_BoxOffice.csv',
    'netflix':    f'{datasets_dir}Daily_Netflix.csv',
    'spy':        f'{datasets_dir}Daily_SPX.csv'
}
# Date format for parsing dates
DATE_FORMAT = '%m/%d/%y'

# Function to load and clean series
def load_series(name):
    """
    Read date & value, parse dates, clean numbers, return a Series
    """
    df = pd.read_csv(
        paths[name],
        usecols=[0, 1],
        header=0,
        dtype={0: str, 1: str}
    )
    df.columns = ['date', name]
    df['date'] = pd.to_datetime(df['date'], format=DATE_FORMAT, errors='coerce')
    df[name] = pd.to_numeric(
        df[name].str.replace(r'[\$,]', '', regex=True),
        errors='coerce'
    )
    df.dropna(subset=['date', name], inplace=True)
    return df.set_index('date')[name].sort_index()

# Load and align level series
daily_box = load_series('box_office')
daily_net = load_series('netflix')
daily_spy = load_series('spy')
df_levels = pd.concat([daily_box, daily_net, daily_spy], axis=1, join='inner')

# Compute log-returns
t_returns = np.log(df_levels).diff().dropna()

if __name__ == '__main__':
    # 1) Cross-correlation on returns (±365 days)
    maxlag = 365
    lags = np.arange(-maxlag, maxlag + 1)
    corrs = [
        t_returns['box_office'].corr(t_returns['netflix'].shift(l))
        for l in lags
    ]

    plt.figure()
    plt.plot(lags, corrs)
    plt.axvline(0, color='black', linewidth=1)
    plt.title('Log-Return Cross-Correlation (±365 days)')
    plt.xlabel('Lag (days; positive = BoxOffice leads)')
    plt.ylabel('Correlation')
    plt.tight_layout()
    plt.savefig('outputs/returns_corr.png')
    plt.close()
    print('➡️  Saved returns correlation plot to outputs/returns_corr.png')

    # 2) Find best positive lag
    pos_mask = lags >= 0
    pos_lags = lags[pos_mask]
    pos_corrs = np.array(corrs)[pos_mask]
    best_idx = int(np.argmax(pos_corrs))
    best_lag = int(pos_lags[best_idx])
    best_corr = pos_corrs[best_idx]
    print(f'\nPeak return correlation = {best_corr:.4f} at lag = {best_lag} days.')

    # 3) OLS regression on returns
    df_reg = t_returns.copy()
    df_reg['box_lag'] = df_reg['box_office'].shift(best_lag)
    df_reg.dropna(subset=['netflix', 'box_lag'], inplace=True)

    X = add_constant(df_reg['box_lag'])
    y = df_reg['netflix']
    model = OLS(y, X).fit()
    print(f'\nOLS Regression (Netflix returns ~ BoxOffice returns lagged {best_lag} days):')
    print(model.summary())

    # 4) Granger causality on returns
    print('\nGranger Causality Tests (BoxOffice → Netflix) on returns:')
    gc_df = t_returns[['netflix', 'box_office']]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        results = grangercausalitytests(gc_df, maxlag=8, verbose=False)
    for lag, res in results.items():
        pval = res[0]['ssr_ftest'][1]
        print(f'  Lag {lag:>2}: p-value = {pval:.4f}')
