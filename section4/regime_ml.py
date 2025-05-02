import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

# ── File: scripts/regime_ml.py ───────────────────────────────────────────────
# Fits a 2-state Markov-switching regression: Netflix returns on lagged BoxOffice returns.

# Configuration
datasets_dir = 'datasets/'
paths = {
    'box_office': f'{datasets_dir}Daily_BoxOffice.csv',
    'netflix':    f'{datasets_dir}Daily_Netflix.csv'
}
DATE_FORMAT = '%m/%d/%y'
best_lag = 6  # from previous returns_analysis

# Function to load and clean series (date,value)
def load_series(name):
    df = pd.read_csv(paths[name], usecols=[0,1], header=0, dtype={0:str,1:str})
    df.columns = ['date', name]
    df['date'] = pd.to_datetime(df['date'], format=DATE_FORMAT, errors='coerce')
    df[name] = pd.to_numeric(df[name].str.replace(r'[\$,]', '', regex=True), errors='coerce')
    df.dropna(subset=['date', name], inplace=True)
    return df.set_index('date')[name].sort_index()

# Load returns
daily_box = load_series('box_office')
daily_net = load_series('netflix')
levels = pd.concat([daily_box, daily_net], axis=1, join='inner')
returns = np.log(levels).diff().dropna()
returns['box_lag'] = returns['box_office'].shift(best_lag)
returns = returns.dropna()

y = returns['netflix']
exog = returns['box_lag']

if __name__ == '__main__':
    # Fit a 2-regime Markov-switching model with switching intercepts and slope
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mod = MarkovRegression(endog=y, k_regimes=2, exog=exog, switching_variance=True)
        res = mod.fit(em_iter=10, search_reps=5)

    print(res.summary())

    # Get smoothed probabilities for regime 1
    prob_regime1 = res.smoothed_marginal_probabilities[0]

    # Plot regime probabilities
    plt.figure()
    plt.plot(prob_regime1.index, prob_regime1.values)
    plt.title('Smoothed Probability of State 1')
    plt.xlabel('Date')
    plt.ylabel('P(State=1)')
    plt.tight_layout()
    plt.savefig('outputs/regime_probs.png')
    plt.close()
    print('➡️  Saved regime probability plot to outputs/regime_probs.png')
