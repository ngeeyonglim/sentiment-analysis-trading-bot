import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Perform Sentiment Analysis on Tweets ---
# Load the processed tweets
tweets_file = 'processed_tweets.csv'
df = pd.read_csv(tweets_file)
df['neg']=0
df['pos']=0
df['neu']=0
df['compound']=0


# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Analyze sentiment for each tweet
for index, row in df.iterrows():
    tweet = row['Tweets']
    sentiment = analyzer.polarity_scores(tweet)
    
    # Store the sentiment scores in the DataFrame
    df.at[index, 'neg'] = sentiment['neg']
    df.at[index, 'pos'] = sentiment['pos']
    df.at[index, 'neu'] = sentiment['neu']
    df.at[index, 'compound'] = sentiment['compound']
    

# Save the results to a new CSV file
output_file = '_3_tweets_with_sentiment.csv'
df.to_csv(output_file, index=False)

print(f"Sentiment analysis completed. Results saved to {output_file}.")




