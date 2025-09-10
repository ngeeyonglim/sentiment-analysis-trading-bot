import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

import pandas as pd

def add_technical_features(df):
    
    # Sort by date just in case
    df = df.sort_values('timestamp')

    # Daily returns
    df['returns'] = df['close'].pct_change()

    # Bollinger Bands (20-day window)
    df['middle_band'] = df['close'].rolling(window=20).mean()
    df['rolling_std'] = df['close'].rolling(window=20).std()
    df['upper_band'] = df['middle_band'] + 2 * df['rolling_std']
    df['lower_band'] = df['middle_band'] - 2 * df['rolling_std']

    # Price position within the bands (0 = lower, 1 = upper)
    df['price_position'] = (df['close'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])

    # Band width as percentage of middle band
    df['band_width'] = (df['upper_band'] - df['lower_band']) / df['middle_band']

    # Momentum (difference between today and 10 days ago)
    df['momentum'] = df['close'] - df['close'].shift(10)

    # Future return (target variable)
    df['future_return'] = df['close'].shift(-1) / df['close'] - 1

    # Drop rows with missing values due to rolling windows or shifts
    df.dropna(inplace=True)

    return df

    

# Load dataset
df = pd.read_csv('crypto_prices_historical.csv')
back_test_df = pd.read_csv('crypto_prices_recent.csv')

df = add_technical_features(df)
back_test_df = add_technical_features(back_test_df)

# Features to use
features = [
    'open', 'high', 'low', 'close', 'returns', 'volume',
    'middle_band', 'rolling_std', 'upper_band', 'lower_band',
    'price_position', 'band_width', 'momentum'
]

# Normalize features
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Prepare time-series input
window_size = 120
X = []
y = []
X_backtest = []

for i in range(window_size, len(df)):
    X.append(df[features].iloc[i - window_size:i].values)
    y.append(df['future_return'].iloc[i])  # <-- use actual future return


X = np.array(X)
y = np.array(y)

print(f"X shape: {X.shape}, y shape: {y.shape}")

# Flatten for model input
Xflat = X.reshape(X.shape[0], -1)

# Train/validation/test split
X_train, X_temp, y_train, y_temp = train_test_split(Xflat, y, test_size=0.3, random_state=42)  # 70% train, 30% temp
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # Split temp into 50% val, 50% test

# Define the regression model
model = XGBRegressor(
    objective='reg:squarederror',  # for regression
    n_estimators=1024,
    max_depth=4,
    learning_rate=0.03,
    early_stopping_rounds=50,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_set=[(X_val, y_val)],  # Use validation set for evaluation
    eval_metric='rmse',  # Evaluation metric
    verbosity=1,  # Verbosity level
    random_state=42
)

# Train the model with validation set
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],  # Use validation set for evaluation
    verbose=True
)

# Predict future returns on the test set
predicted_returns = model.predict(X_test)
buy_strength = np.clip(predicted_returns, -1, 1)


# Calculate evaluation metrics
mse = mean_squared_error(y_test, buy_strength)
mae = mean_absolute_error(y_test, buy_strength)
r2 = r2_score(y_test, buy_strength)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")

# Check directional accuracy
correct_directions = np.sum(np.sign(buy_strength) == np.sign(y_test))
directional_accuracy = correct_directions / len(y_test) * 100

print(f"Directional Accuracy: {directional_accuracy:.2f}%")

# Baseline R² using the mean of y_test
baseline_predictions = np.full_like(y_test, np.mean(y_test))
baseline_r2 = r2_score(y_test, baseline_predictions)
print(f"Baseline R²: {baseline_r2}")

# Backtesting strategy

def backtest_strategy(df, buy_strength):
    """
    Backtest the trading strategy based on buy strength. 
    Buy strength determines how much bitcoin to buy or sell.
    Buy up to 1 bitcoin of buy strength value. If buy strength is negative, sell up to 1 bitcoin.
    

    Parameters:
    - df: DataFrame containing the original data.
    - buy_strength: Array of predicted buy strength values.
    
    Returns:
    - DataFrame with backtesting results.
    """
    print("SHAPE CHECK")
    print(df.shape)
    print(buy_strength.shape)

    df['buy_strength'] = buy_strength
    df['position'] = 0  # Position in bitcoin (in BTC)
    df['cash'] = 0  # Starting cash in USD
    df['holdings'] = 0  # Value of holdings in USD
    df['total_value'] = 0

    starting_cash = 200000.0  # Starting cash in USD
    starting_position = 0.0


    for i in range(len(df)):
        if df['buy_strength'].iloc[i] > 0:
            # Buy up to 1 bitcoin if buy strength is positive
            buy_amount = df['buy_strength'].iloc[i]
            if (buy_amount * df['close'].iloc[i]) <= starting_cash:
                starting_position += buy_amount
                starting_cash -= buy_amount * df['close'].iloc[i]
            else:
                # If not enough cash, buy as much as possible
                max_buyable = df['cash'].iloc[i] / df['close'].iloc[i]
                starting_position += max_buyable
                starting_cash -= max_buyable * df['close'].iloc[i]
        elif df['buy_strength'].iloc[i] < 0:
            # Sell up to 1 bitcoin if buy strength is negative
            sell_amount = -df['buy_strength'].iloc[i]
            if sell_amount <= starting_position:
                starting_position -= sell_amount
                starting_cash += sell_amount * df['close'].iloc[i]
            else:  
                # If not enough position, sell all
                starting_cash += starting_position * df['close'].iloc[i]
                starting_position = 0.0
        
        # Update holdings and total value
        df.at[i, 'position'] = starting_position
        df.at[i, 'cash'] = starting_cash
        df.at[i, 'holdings'] = df.at[i, 'position'] * df.at[i, 'close']
        df.at[i, 'total_value'] = df.at[i, 'cash'] + df.at[i, 'holdings']

    return df

for i in range(window_size, len(back_test_df)):
    X_backtest.append(back_test_df[features].iloc[i - window_size:i].values)

X_backtest = np.array(X_backtest)

# Flatten for model input
X_backtest_flat = X_backtest.reshape(X_backtest.shape[0], -1)

back_test_predicted_buy_strength = model.predict(X_backtest_flat)
back_test_predicted_buy_strength = np.clip(back_test_predicted_buy_strength, -1, 1)  # Clip to [-1, 1] for buy strength
back_test_predicted_buy_strength = back_test_predicted_buy_strength * 100 # Use the same predicted returns for backtesting

# Create a new DataFrame for backtesting that aligns with the predicted returns
df_backtest = back_test_df.iloc[window_size:].copy()  # Exclude the first `window_size` rows
df_backtest.reset_index(drop=True, inplace=True)  # Reset the index to start from 0

# Ensure the timestamp column is included
if 'timestamp' not in df_backtest.columns:
    df_backtest['timestamp'] = df.iloc[window_size:]['timestamp'].values  # Copy the timestamp column

# Add the predicted returns (buy_strength) to the new DataFrame
df_backtest['buy_strength'] = back_test_predicted_buy_strength

# Backtest the strategy using the aligned DataFrame
backtested_df = backtest_strategy(df_backtest, df_backtest['buy_strength'].copy())

print("Backtesting completed. Results:")
#print final total value
print(f"Final Total Value: {backtested_df['total_value'].iloc[-1]:.2f} USD")

# Save backtested results to excel
backtested_df.to_excel('backtested_results.xlsx', index=False)

# # Plot total value and position over time
# plt.figure(figsize=(14, 7))
# plt.plot(backtested_df['timestamp'], backtested_df['total_value'], label='Total Value', color='blue')
# plt.plot(backtested_df['timestamp'], backtested_df['close'], label='Position Value', color='orange')
# plt.xlabel('Date')
# plt.ylabel('Value (USD)')
# plt.title('Total Value and Position Value Over Time')
# plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()  # Automatically adjust layout to fit everything
# plt.show()