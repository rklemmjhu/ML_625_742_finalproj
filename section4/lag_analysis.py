import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.api import OLS, add_constant
from statsmodels.tsa.stattools import grangercausalitytests

# ── File: scripts/lag_analysis.py ─────────────────────────────────────────────
# Scans BoxOffice vs. Netflix lead/lag up to ±1460 days,
# finds the best positive lag, runs OLS and Granger tests.

# Paths for uniform two-column CSVs
datasets_dir = 'datasets/'
paths = {
    'box_office': f'{datasets_dir}Daily_BoxOffice.csv',
    'netflix':    f'{datasets_dir}Daily_Netflix.csv',
    'spy':        f'{datasets_dir}Daily_SPX.csv',
    'gdp':        f'{datasets_dir}Quarterly_GDP.csv'
}
# Date format in all CSVs (MM/DD/YY)
DATE_FORMAT = '%m/%d/%y'


def load_series(name):
    """
    Read date + value from CSV, parse dates, clean numeric, drop bad rows,
    return a pandas Series indexed by date.
    """
    df = pd.read_csv(
        paths[name], usecols=[0,1], header=0,
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

# Load daily series
daily_box = load_series('box_office')
daily_net = load_series('netflix')
daily_spy = load_series('spy')
# Quarterly GDP (not used in this script)
# quarterly_gdp = load_series('gdp')

# Align on common dates (inner join)
daily = pd.concat([daily_box, daily_net, daily_spy], axis=1, join='inner')


def main():
    # Part 1: Cross-correlation over ±1460 days
    maxlag = 1460
    lags = np.arange(-maxlag, maxlag + 1)
    corrs = [
        daily['box_office'].corr(daily['netflix'].shift(l))
        for l in lags
    ]

    plt.figure()
    plt.plot(lags, corrs)
    plt.axvline(0, color='black', linewidth=1)
    plt.xlabel('Lag (days; positive = BoxOffice leads)')
    plt.ylabel('Correlation')
    plt.title(f'Box Office vs Netflix Cross-Correlation (±{maxlag} days)')
    plt.tight_layout()
    plt.savefig('outputs/box_net_corr.png')
    plt.close()
    print(f"\n➡️  Saved cross-correlation plot to outputs/box_net_corr.png")

    # Part 2: Identify best positive lag
    pos_mask = lags >= 0
    pos_lags = lags[pos_mask]
    pos_corrs = np.array(corrs)[pos_mask]
    best_idx = int(np.argmax(pos_corrs))
    best_lag = int(pos_lags[best_idx])
    best_corr = pos_corrs[best_idx]
    print(f"\nPeak correlation = {best_corr:.4f} at lag = {best_lag} days.")

    # Part 3: OLS regression at best lag
    df_reg = daily.copy()
    df_reg['box_lag'] = df_reg['box_office'].shift(best_lag)
    df_reg.dropna(subset=['netflix', 'box_lag'], inplace=True)

    X = add_constant(df_reg['box_lag'])
    y = df_reg['netflix']
    model = OLS(y, X).fit()
    print(f"\nOLS Regression (Netflix ~ BoxOffice lagged {best_lag} days):")
    print(model.summary())

    # Part 4: Granger causality (BoxOffice → Netflix)
    print("\nGranger Causality Tests (BoxOffice → Netflix):")
    gc_df = daily[['netflix', 'box_office']].dropna()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        results = grangercausalitytests(gc_df, maxlag=8, verbose=False)
    for lag, res in results.items():
        pval = res[0]['ssr_ftest'][1]
        print(f"  Lag {lag:>2}: p-value = {pval:.4f}")

if __name__ == '__main__':
    main()
