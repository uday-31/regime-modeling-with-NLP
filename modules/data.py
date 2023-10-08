from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf


def get_ts_data(tickers, start_date, delta, end_date='2022-12-31'):
    data = yf.download(tickers, start=start_date, end=end_date)  # time series data
    df_ts_close = data['Adj Close'].dropna()
    df_ts_open = data['Open'].dropna()
    df_ts_open.index = df_ts_open.index + pd.Timedelta(f'{delta}h')  # adjust time
    df_ts = pd.concat([df_ts_close, df_ts_open]).sort_index()
    
    return df_ts


def get_text_data(fpath='../assets/fomc_documents.csv', s_date=datetime(1985, 1, 1)):
    # Get starting year for text data
    if isinstance(s_date, datetime):
        s_year = s_date.date().year
    elif isinstance(s_date, str) or isinstance(s_date, int):
        s_year = int(str(s_date)[:4])
    else:
        print(f"Check that your input date is in the correct format!")
        s_year = None
    s_year = s_year if s_year is not None else 1985
    
    # Read in FOMC Data
    try:
        fomc_data = pd.read_csv(fpath)
    except Exception as e:
        print(f"Please check that {fpath} is a valid path pointing to FOMC CSV data!")
        return
    
    fomc_data.meeting_date = pd.to_datetime(fomc_data.meeting_date)
    fomc_data = fomc_data[fomc_data.document_kind.isin([
        'historical_minutes',
        'minutes',
        'minutes_of_actions'
    ])]
    # fomc_data['meeting_year'] = fomc_data.meeting_date.dt.year
    fomc_data = fomc_data[fomc_data.meeting_date.dt.year >= s_year]
    return fomc_data


if __name__ == "__main__":
    print(f"Please import this file as a module!")
