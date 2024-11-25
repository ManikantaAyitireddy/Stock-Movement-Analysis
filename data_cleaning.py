import pandas as pd

# Load CSV file
df_investing = pd.read_csv('invest_data.csv')
df_stocks= pd.read_csv('stock_data.csv')
# Check column names
print(df_investing.columns)

# Combine the two datasets
df = pd.concat([df_investing, df_stocks], ignore_index=True)



# Inspect the dataset
print(df.shape)  # Check the number of rows and columns
print(df.columns)  # See all the column names
print(df.head())  # Display the first few rows
# Check for missing values in the dataset

df['text'] = df['text'].fillna('missing')
df['cleaned_text'] = df['cleaned_text'].fillna('missing')
print(df.isnull().sum())
# Drop rows with missing cleaned_text or text_sentiment
df = df.dropna(subset=['cleaned_text', 'text_sentiment'])

df['cleaned_title'] = df['cleaned_title'].fillna('missing')

print(df.isnull().sum())

# Check the combined dataframe
print(df.shape)  # Should show 800 rows (600 + 200)
print(df.head())
df.to_csv(r'C:/Users/bikki/Downloads/combined_data.csv', index=False)
