import praw

# Replace these with your own credentials from Reddit App
client_id = 'ycJvtsxJig0l4D6-vLQflw'        # Example: 'my_client_id'
client_secret = 'RZG7jsWwIbGoHdr8Y3doYfrtiahbCQ' # Example: 'my_client_secret'
user_agent = 'my_scraper'       # Example: 'my_scraper'

# Initialize the Reddit instance
reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent)
# Choose the subreddit
subreddit = reddit.subreddit('investments')  # You can change 'stocks' to any subreddit
# subreddit = reddit.subreddit('stocks')

# Fetch the top 1000 hot posts
for submission in subreddit.hot(limit=1000):
    print(f"Title: {submission.title}")
    print(f"URL: {submission.url}")
    print(f"Upvotes: {submission.score}")
    print(f"Text: {submission.selftext}")
    print("-" * 80)
import re

# Function to clean the text
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions (e.g., @username)
    text = re.sub(r'@\w+', '', text)
    # Remove special characters and numbers, keeping only alphabetic characters
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

# Apply the cleaning function to each post's title and selftext
for submission in subreddit.hot(limit=1000):
    title_cleaned = clean_text(submission.title)
    text_cleaned = clean_text(submission.selftext)
    
    print(f"Cleaned Title: {title_cleaned}")
    print(f"Cleaned Text: {text_cleaned}")
    print("-" * 80)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get sentiment
def get_sentiment(text):
    sentiment_score = analyzer.polarity_scores(text)
    return sentiment_score['compound']  # The 'compound' score is a normalized score between -1 and 1

# Apply sentiment analysis to each post's title and selftext
for submission in subreddit.hot(limit=1000):
    title_cleaned = clean_text(submission.title)
    text_cleaned = clean_text(submission.selftext)
    
    title_sentiment = get_sentiment(title_cleaned)
    text_sentiment = get_sentiment(text_cleaned)
    
    print(f"Title Sentiment Score: {title_sentiment}")
    print(f"Text Sentiment Score: {text_sentiment}")
    print("-" * 80)
import pandas as pd

# Create a list to store the data
data = []

# Collect the data
for submission in subreddit.hot(limit=1000):
    title_cleaned = clean_text(submission.title)
    text_cleaned = clean_text(submission.selftext)
    
    title_sentiment = get_sentiment(title_cleaned)
    text_sentiment = get_sentiment(text_cleaned)
    
    data.append({
        'title': submission.title,
        'cleaned_title': title_cleaned,
        'title_sentiment': title_sentiment,
        'text': submission.selftext,
        'cleaned_text': text_cleaned,
        'text_sentiment': text_sentiment,
        'upvotes': submission.score,
        'url': submission.url
    })

# Create a DataFrame from the collected data
df = pd.DataFrame(data)

# Display the first few rows
print(df.head())
# Save the DataFrame to a CSV file
df.to_csv('reddit_stock_data.csv', index=False)

print("Data saved to 'reddit_stock_data.csv'")
# Save to a specific directory
df.to_csv(r'C:/Users/bikki/Downloads/reddit_stock_data6.csv', index=False)
