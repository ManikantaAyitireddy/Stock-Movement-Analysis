Stock Movement Prediction Using Social Media Sentiment
This project aims to predict stock price movements using sentiment analysis of social media data. The workflow includes web scraping, data cleaning, sentiment analysis, feature engineering, and machine learning.

Project Structure
project/
│
├── data/
│   ├── invest_data.csv               # Dataset 1: Raw social media data from investments
│   ├── stock_data.csv                # Dataset 2: Raw social media data from stocks
│   ├── combined_data.csv             # Cleaned and combined dataset
│
├── models/
│   ├── sentiment_stock_model.pkl     # Trained logistic regression model
│
├── scripts/
│   ├── scrape_reddit.py              # Script for scraping Reddit data
│   ├── data_cleaning.py              # Script for cleaning and preprocessing data
│   ├── model_training.py             # Script for sentiment analysis and model training
│
├── README.md                         # Project documentation
└── requirements.txt                  # Python dependencies

Steps to Run the Project
1. Web Scraping
Run the scrape_reddit.py script to scrape data from Reddit. This script collects social media posts, including titles, selftext, and upvotes, from specified subreddits.

Output: Raw data saved as invest_data.csv and stock_data.csv in the data/ folder.
Command to Run:
python scripts/scrape_reddit.py

2. Data Cleaning
Run data_cleaning.py to preprocess and clean the scraped data. It handles:

Removing noise (e.g., URLs, mentions, special characters).
Filling missing values and dropping unnecessary rows.
Combining data into a single dataset, combined_data.csv.
Command to Run:
python scripts/data_cleaning.py

3. Sentiment Analysis & Model Training
Run model_training.py to perform sentiment analysis, create features, and train the machine learning model. It includes:

Analyzing sentiment using VADER sentiment analysis.
Engineering features such as engagement_score and overall_sentiment.
Training a logistic regression model to predict stock movements.
Saving the trained model as sentiment_stock_model.pkl in the models/ folder.
Command to Run:

bash
Copy code
python scripts/model_training.py

Dependencies

Ensure all required Python libraries are installed. Use the following command to install dependencies:

Command to Run:

bash
Copy code
pip install -r requirements.txt

Key Libraries Used:

praw (for web scraping from Reddit)
pandas (for data manipulation)
numpy (for numerical computations)
vaderSentiment (for sentiment analysis)
scikit-learn (for machine learning)
joblib (for saving the trained model)

Outputs

Data Files:
invest_data.csv and stock_data.csv: Raw data scraped from Reddit.
combined_data.csv: Cleaned and processed dataset ready for analysis.
Model:
sentiment_stock_model.pkl: Trained logistic regression model.

Metrics:
Accuracy: 99%
Precision: 100%
Recall: 99%

How It Works
Web Scraping:

Scrapes Reddit posts from relevant subreddits about stock market discussions.

Data Cleaning:

Cleans and preprocesses raw data for analysis.

Sentiment Analysis:

Computes sentiment scores (Positive, Negative, Neutral) using VADER.

Feature Engineering:

Features like engagement_score and overall_sentiment are created.

Prediction:

Logistic regression predicts stock movements based on sentiment and engagement.

Future Enhancements
Extend to other platforms like Twitter or Telegram for more data.
Use advanced NLP models like transformers for more accurate sentiment analysis.
Implement a real-time stock movement prediction dashboard.

