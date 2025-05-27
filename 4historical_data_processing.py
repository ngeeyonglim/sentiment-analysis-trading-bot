import pandas as pd

# File paths
input_file = 'btcusd_1-min_data.csv'
output_file = 'crypto_prices_historical.csv'

# Prepare the output file with headers
first_write = True

# Dictionary to accumulate one day's data
current_day = None
day_data = []

# Read in chunks
chunk_iter = pd.read_csv(input_file, chunksize=100_000)


for chunk in chunk_iter:

    chunk.rename(columns={
        'Timestamp': 'timestamp',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)

    # Ensure timestamps are datetime
    chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], unit='s')

    # Sort chunk just in case
    chunk = chunk.sort_values('timestamp')

    # Extract date only
    chunk['timestamp'] = chunk['timestamp'].dt.date

    for date, group in chunk.groupby('date'):
        if current_day is None:
            current_day = date

        if date != current_day:
            # Process the previous day's data
            df_day = pd.DataFrame(day_data)
            o = df_day.iloc[0]['open']
            h = df_day['high'].max()
            l = df_day['low'].min()
            c = df_day.iloc[-1]['close']
            v = df_day['volume'].sum()

            # Append row to output file
            daily_row = pd.DataFrame([[current_day, o, h, l, c, v]],
                                     columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            daily_row.to_csv(output_file, mode='a', header=first_write, index=False)
            first_write = False

            # Reset for new day
            current_day = date
            day_data = []

        # Accumulate current row
        day_data.extend(group.to_dict(orient='records'))

# Final day's data
if day_data:
    df_day = pd.DataFrame(day_data)
    o = df_day.iloc[0]['open']
    h = df_day['high'].max()
    l = df_day['low'].min()
    c = df_day.iloc[-1]['close']
    v = df_day['volume'].sum()
    daily_row = pd.DataFrame([[current_day, o, h, l, c, v]],
                             columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    daily_row.to_csv(output_file, mode='a', header=first_write, index=False)
