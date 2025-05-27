import requests
import pandas as pd

def get_crypto_prices_x_days(x=365, currency='USD'):
    """
    Fetches cryptocurrency OHLC (Open, High, Low, Close) prices, returns, and volume for the last x months.
    
    Parameters:
    x (int): Number of months to fetch data for.
    currency (str): Currency in which to fetch prices.
    
    Returns:
    DataFrame: A DataFrame containing the OHLC prices, returns, and volume.
    """
    url = f'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency={currency}&days={x}&interval=daily'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        
        # Extract OHLC and volume data
        prices = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])
        market_caps = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])
        total_volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
        
        # Merge data into a single DataFrame
        prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
        market_caps['timestamp'] = pd.to_datetime(market_caps['timestamp'], unit='ms')
        total_volumes['timestamp'] = pd.to_datetime(total_volumes['timestamp'], unit='ms')
        
        df = prices.merge(total_volumes, on='timestamp', how='left')
        df = df.merge(market_caps, on='timestamp', how='left')
        
        # Calculate OHLC (Open, High, Low, Close) from the 'close' prices
        df['open'] = df['close'].shift(1)  # Open is the previous day's close
        df['high'] = df['close'].rolling(2).max()  # High is the max of the last 2 days
        df['low'] = df['close'].rolling(2).min()  # Low is the min of the last 2 days
        
        # Reorder columns for clarity
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume',]]
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        return df
    else:
        raise Exception(f"Error fetching data: {response.status_code}")

df = get_crypto_prices_x_days(365, 'USD')

# save the DataFrame to a CSV file
output_file = 'crypto_prices_recent.csv'
df.to_csv(output_file, index=False)
print(f"Cryptocurrency prices saved to {output_file}.")

