import pandas as pd
from binance.client import Client


client = Client()


def get_data(ticker):
    df = pd.DataFrame(
        client.get_historical_klines(
            ticker,
            '1m',
            '2023-10-17',
            '2023-10-21'
        )
    )
    df = df.iloc[:, :6]
    df.columns = [
        'Time',
        'Open',
        'High',
        'Low',
        'Close',
        'Volume'
    ]
    df.set_index('Time', inplace=True)
    df.index = pd.to_datetime(df.index, unit='ms')
    return df.astype(float)


get_data('BTCUSDT').to_csv('data/raw/btcusdt.csv')
