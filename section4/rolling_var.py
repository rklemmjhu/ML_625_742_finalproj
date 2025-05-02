import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

# ── File: scripts/rolling_var.py ─────────────────────────────────────────────
# Estimates a rolling-window VAR on daily returns and tracks
# the 1-day impulse-response of Netflix to a BoxOffice shock.

# Suppress frequency-related warnings via message matching
warnings.filterwarnings(
    "ignore",
    message=".*no associated frequency information.*"
)

# Configuration
datasets_dir = 'datasets/'
paths = {
    'box_office': f'{datasets_dir}Daily_BoxOffice.csv',
    'netflix':    f'{datasets_dir}Daily_Netflix.csv',
    'spy':        f'{datasets_dir}Daily_SPX.csv',
}
DATE_FORMAT = '%m/%d/%y'
window_size = 252        # trading days per year
var_lags = 1             # VAR(p) lag order
irf_horizon = 1          # 1-day ahead impulse response

# Function to load and clean series
def load_series(name):
    df = pd.read_csv(
        paths[name], usecols=[0,1], header=0,
        dtype={0: str, 1: str}
    )
    df.columns = ['date', name]
    df['date'] = pd.to_datetime(df['date'], format=DATE_FORMAT, errors='coerce')
    df[name] = pd.to_numeric(
        df[name].str.replace(r'[\$,]', '', regex=True), errors='coerce'
    )
    df.dropna(subset=['date', name], inplace=True)
    return df.set_index('date')[name].sort_index()

# Load level series and compute log-returns
daily_box = load_series('box_office')
daily_net = load_series('netflix')
daily_spy = load_series('spy')
levels = pd.concat([daily_box, daily_net, daily_spy], axis=1, join='inner')
returns = np.log(levels).diff().dropna()

if __name__ == '__main__':
    irf_series = []
    irf_dates = []

    # Rolling-window VAR
    for end in range(window_size, len(returns)):
        window_data = returns.iloc[end-window_size:end]
        model = VAR(window_data)
        result = model.fit(var_lags)

        # 1-day ahead IRF: Netflix response to unit BoxOffice shock
        irf = result.irf(irf_horizon)
        coeff = irf.irfs[irf_horizon,
                         window_data.columns.get_loc('netflix'),
                         window_data.columns.get_loc('box_office')]

        irf_series.append(coeff)
        irf_dates.append(returns.index[end])

    # Convert to DataFrame and plot
    df_irf = pd.Series(irf_series, index=irf_dates)
    plt.figure()
    plt.plot(df_irf.index, df_irf.values)
    plt.title('Rolling 1-Day IRF: Netflix response to BoxOffice shock')
    plt.xlabel('Date')
    plt.ylabel('Impulse-Response Coefficient')
    plt.tight_layout()
    plt.savefig('outputs/rolling_irf.png')
    plt.close()
    print('➡️  Saved rolling IRF plot to outputs/rolling_irf.png')
