import pandas as pd


# Function to add technical indicators
def add_technical_indicators(df):
    # Calculate MACD
    df['ema12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']

    # Calculate Force Index
    df['force_index'] = df['Close'].diff(1) * df['Volume']


data = pd.read_csv('data/raw/btcusdt.csv')
add_technical_indicators(data)
data.to_csv('data/processed/btcusdt.csv')
