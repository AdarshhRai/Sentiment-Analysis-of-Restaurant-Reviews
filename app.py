# BY ADARSH RAI , P.No. : 9326567054 , AICTE no. : STU64c4045d9a6111690567773 , Email ID : - iamadarshrai34@gmail.com

from flask import Flask, render_template, send_from_directory

app = Flask(__name__)

@app.route('/')
def sentiment_analysis():
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

    # COLORS MAPPING
    color_mapping = {0: 'green', 1: 'red'}

    # Visualization graph
    df['sentiment'].map(color_mapping).value_counts().plot(kind='bar', color=df['sentiment'].map(color_mapping))
    plt.xlabel('SENTIMENT OF THE CUSTOMERS ')
    plt.ylabel('TOTAL NO. OF REVIEWS ')
    plt.title('SENTIMENT ANALYSIS OF RESTAURANT REVIEWS: ')
    plt.xticks([0, 1], ['NEGATIVE ->', 'POSITIVE -> '])
    plt.savefig('static/sentiment_bar_chart.png')  # Save the bar chart as an image
    plt.close()

    return render_template('sentiment_analysis.html', accuracy=accuracy, classification_report=classification_report(y_test, y_pred))

@app.route('/static/<filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
