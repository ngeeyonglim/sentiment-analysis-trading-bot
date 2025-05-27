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
        
        # Calculate daily returns
        df['returns'] = df['close'].diff()  # Today's close - Yesterday's close
        
        # Reorder columns for clarity
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'returns', 'volume', 'market_cap']]
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'returns', 'volume', 'market_cap']
        
        return df
    else:
        raise Exception(f"Error fetching data: {response.status_code}")

df = get_crypto_prices_x_days(365, 'USD')

# Calculate rolling mean and standard deviation for volatility
df['middle_band'] = df['close'].rolling(window=30).mean()  # 30-day rolling mean
df['rolling_std'] = df['close'].rolling(window=30).std()  # 30-day rolling standard deviation

# Calculate Bollinger Bands
df['upper_band'] = df['middle_band'] + (df['rolling_std'] * 2)  # Upper Bollinger Band
df['lower_band'] = df['middle_band'] - (df['rolling_std'] * 2)  # Lower Bollinger Band

# Price position
df['price_position'] = (df['close'] - df['middle_band']) / (df['upper_band'] - df['middle_band'])  # Normalized price position

df['band_width'] = df['upper_band'] - df['lower_band'] 

df['volume_to_market_cap'] = df['volume'] / df['market_cap']  

df['momentum'] = df['close'].pct_change(periods=7)  # 14-day momentum



# save the DataFrame to a CSV file
output_file = 'crypto_prices.csv'
df.to_csv(output_file, index=False)
print(f"Cryptocurrency prices saved to {output_file}.")

