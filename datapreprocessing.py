import pandas as pd
import os

df = pd.read_excel('tweets.xlsx')

print(df.head())

# Drop duplicates
df = df.drop_duplicates(subset=['Tweets'])

# Convert to lowercase
df['Tweets'] = df['Tweets'].str.lower()

# delete old processed file if it exists
if os.path.exists('processed_tweets.xlsx'):
    os.remove('processed_tweets.xlsx')

# Save the processed DataFrame to a new csv file
df.to_csv('processed_tweets.csv', index=False)


