import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


DATASETS_DIR = 'datasets/'
PATHS = {
    'boxoffice': f'{DATASETS_DIR}Daily_BoxOffice.csv',
    'netflix' : f'{DATASETS_DIR}Daily_Netflix.csv',
    'spx'     : f'{DATASETS_DIR}Daily_SPX.csv',
    'gdp'     : f'{DATASETS_DIR}Quarterly_GDP.csv'
}

DATE_FORMAT = '%m/%d/%y'  


# ── HELPERS ───────────────────────────────────────────────────────────────────
def load_daily(csv_path, value_name):
    """
    Read a two-column daily CSV (Date, Value),
    strip $/commas, parse numeric,
    return a pd.Series indexed by datetime.
    """
    df = pd.read_csv(
        csv_path,
        usecols=[0,1],
        names=['Date', value_name],
        header=0,
        dtype={0: str, 1: str},
    )
    df['Date'] = pd.to_datetime(df['Date'], format=DATE_FORMAT, errors='coerce')
    df[value_name] = (
        df[value_name]
        .str.replace(r'[\$,]', '', regex=True)
        .astype(float, errors='ignore')
    )
    df = df.dropna(subset=['Date', value_name])
    return df.set_index('Date')[value_name].sort_index()


def load_quarterly_gdp(csv_path):
    """
    Read your quarterly GDP CSV (Date, GDP),
    strip $/commas, parse numeric,
    return a pd.Series indexed quarterly.
    """
    df = pd.read_csv(
        csv_path,
        usecols=[0,1],
        names=['Date','gdp_q'],
        header=0,
        dtype={0:str,1:str},
    )
    df['Date'] = pd.to_datetime(df['Date'], format=DATE_FORMAT, errors='coerce')
    df['gdp_q'] = (
        df['gdp_q']
        .str.replace(r'[\$,]', '', regex=True)
        .astype(float, errors='ignore')
    )
    df = df.dropna(subset=['Date','gdp_q'])
    # take the last known value in each calendar quarter
    return df.set_index('Date')['gdp_q'].resample('Q').last()


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    # 1) Load daily flows & prices
    bo = load_daily(PATHS['boxoffice'], 'boxoffice')
    nf = load_daily(PATHS['netflix'],  'netflix')
    spx= load_daily(PATHS['spx'],      'spx')

    # 2) Aggregate to quarterly
    bo_q  = bo .resample('Q').sum()   # flows: sum
    nf_q  = nf .resample('Q').last()  # price: end-of-period
    spx_q = spx.resample('Q').last()  # price: end-of-period

    # 3) Load quarterly GDP
    gdp_q = load_quarterly_gdp(PATHS['gdp'])

    # 4) Merge into one table
    df = pd.concat([bo_q, nf_q, spx_q, gdp_q], axis=1, join='inner')
    df.columns = ['boxoffice_q','netflix_q','spx_q','gdp_q']

    # 5) Interaction term
    df['interaction'] = df['boxoffice_q'] * df['gdp_q']

    # 6) Fit OLS
    X = df[['boxoffice_q','spx_q','gdp_q','interaction']]
    X = sm.add_constant(X)
    y = df['netflix_q']

    model = sm.OLS(y, X).fit()
    print(model.summary())

    # 7) Plot Actual vs Fitted
    df['fitted'] = model.predict(X)
    plt.figure(figsize=(8,5))
    plt.plot(df.index, df['netflix_q'], label='Actual')
    plt.plot(df.index, df['fitted'],    label='Fitted')
    plt.legend()
    plt.title('Quarterly Netflix: Actual vs Fitted')
    plt.xlabel('Quarter')
    plt.ylabel('Netflix (USD)')
    plt.tight_layout()
    plt.savefig('outputs/quarterly_fit.png')
    print('✔ Saved quarterly fit plot to outputs/quarterly_fit.png')


if __name__ == '__main__':
    main()
