import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# READING THE XLSX FILES
file_path = 'Restaurant_reviews.xlsx'
df = pd.read_excel(file_path)

# FIRST FEW LINES OF THE DATASET
print("\n\n First few rows of the Dataset are as follows :- \n\n")
print(df.head())

# USING VADER FOR ANALYSIS
analyzer = SentimentIntensityAnalyzer()
df['sentiment'] = df['Review'].apply(lambda x: 1 if analyzer.polarity_scores(x)['compound'] >= 0 else 0)

# SPLITTING THE DATA
X = df['Review']
y = df['Liked']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# TRAINING THE DATA
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# PREDICTION
y_pred = classifier.predict(X_test_tfidf)

# ACCURACY
accuracy = accuracy_score(y_test, y_pred)
print("\n\n Now we will have the accuracy for this model :- \n\n"f'Accuracy: {accuracy:.2f}'"\n\n")
print(classification_report(y_test, y_pred))

# Create color mapping for sentiment
color_mapping = {0: 'green', 1: 'red'}

# VISUALIZATION GRAPH
df['sentiment'].map(color_mapping).value_counts().plot(kind='bar', color=df['sentiment'].map(color_mapping))
plt.xlabel('SENTIMENT OF THE CUSTOMERS ')
plt.ylabel('TOTAL NO. OF REVIEWS ')
plt.title('SENTIMENT ANALYSIS OF RESTAURANT REVIEWS: ')
plt.xticks([0, 1], ['NEGATIVE ->', 'POSITIVE -> '])
plt.show()
