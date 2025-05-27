def get_crypto_intraday_prices(days=30, currency='USD'):
    """
    Fetches cryptocurrency intraday prices (hourly) for the last `days` days.
    
    Parameters:
    days (int): Number of days to fetch intraday data for (max 90 days).
    currency (str): Currency in which to fetch prices.
    
    Returns:
    DataFrame: A DataFrame containing intraday prices and volume.
    """
    if days > 90:
        days = 90  # CoinGecko API limits to 90 days for hourly data
    if days < 1:
        raise ValueError("Days must be at least 1.")
    
    headers = {
        'User-Agent': 'Mozilla/5.0'
    }

    url = f'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency={currency}&days={days}&interval=hourly'
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        
        # Extract hourly prices and volumes
        prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
        
        # Convert timestamps to datetime
        prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
        volumes['timestamp'] = pd.to_datetime(volumes['timestamp'], unit='ms')
        
        # Merge prices and volumes into a single DataFrame
        df = prices.merge(volumes, on='timestamp', how='left')
        
        # Calculate intraday metrics
        df['hour'] = df['timestamp'].dt.hour
        df['date'] = df['timestamp'].dt.date
        
        # Group by date to calculate daily open, high, low, close (OHLC)
        daily_ohlc = df.groupby('date').agg({
            'price': ['first', 'max', 'min', 'last'],  # Open, High, Low, Close
            'volume': 'sum'  # Total volume for the day
        }).reset_index()
        daily_ohlc.columns = ['date', 'open', 'high', 'low', 'close', 'daily_volume']
        
        # Calculate intraday returns
        df['intraday_return'] = df['price'].pct_change()  # Percentage change between consecutive prices
        
        # Calculate intraday deviations (difference from daily average price)
        daily_avg_price = df.groupby('date')['price'].transform('mean')
        df['intraday_deviation'] = df['price'] - daily_avg_price
        
        # Calculate intraday volatility (rolling standard deviation)
        df['intraday_volatility'] = df['price'].rolling(window=24).std()  # 24-hour rolling volatility
        
        return df, daily_ohlc
    else:
        raise Exception(f"Error fetching data: {response.status_code}")

# Fetch intraday data for the last 30 days
intraday_df, daily_ohlc_df = get_crypto_intraday_prices(days=30, currency='USD')

# Save the intraday and daily OHLC data to CSV files
intraday_output_file = 'crypto_intraday_prices.csv'
daily_ohlc_output_file = 'crypto_daily_ohlc.csv'

intraday_df.to_csv(intraday_output_file, index=False)
daily_ohlc_df.to_csv(daily_ohlc_output_file, index=False)

print(f"Intraday prices saved to {intraday_output_file}.")
print(f"Daily OHLC data saved to {daily_ohlc_output_file}.")