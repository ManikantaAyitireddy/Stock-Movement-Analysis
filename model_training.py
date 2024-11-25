import pandas as pd

# Load the combined dataset
df = pd.read_csv('combined_data.csv')

print(df[['cleaned_title', 'title_sentiment', 'cleaned_text', 'text_sentiment']].head(10))
df['engagement_score'] = df['upvotes'] * (df['title_sentiment'] + df['text_sentiment'])
df['overall_sentiment'] = (df['title_sentiment'] + df['text_sentiment']) / 2
def sentiment_category(score):
    if score > 0.2:
        return 'Positive'
    elif score < -0.2:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment_category'] = df['overall_sentiment'].apply(sentiment_category)
# Convert sentiment category to binary labels (1 for Positive, 0 for Neutral/Negative)
df['stock_movement'] = df['sentiment_category'].apply(lambda x: 1 if x == 'Positive' else 0)
y = df['stock_movement']
X = df[['upvotes', 'overall_sentiment', 'engagement_score']]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
# Initialize the Logistic Regression model
model = LogisticRegression(random_state=42)

# Train the model
model.fit(X_train, y_train)
print("Model training completed!")
# Predict on test data
y_pred = model.predict(X_test)
# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Print the evaluation results
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Testing Accuracy: {test_accuracy:.2f}")
import numpy as np

feature_importance = np.abs(model.coef_[0])
feature_names = X.columns
feature_importance_dict = dict(zip(feature_names, feature_importance))

# Sort by importance
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
print("Feature Importance:")
for feature, importance in sorted_features:
    print(f"{feature}: {importance:.4f}")

import joblib

# Save the model to a file
joblib.dump(model, 'sentiment_stock_model.pkl')
print("Model saved as 'sentiment_stock_model.pkl'")

