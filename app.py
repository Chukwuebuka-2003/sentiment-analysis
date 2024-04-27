import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the saved model
loaded_model = joblib.load('sentiment_analysis_model.sav')

tfidf_vectorizer = TfidfVectorizer(max_features=1000)  

df = pd.read_csv('Expanded_IPPIS_Opinions_Dataset.csv')

# Split data into features (X) and target variable (y)
X = df['Opinion']
y = df['Impact']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)



# Function to preprocess input text
def preprocess_text(text):
    # Preprocess text here if needed
    return text

# Function to predict sentiment
def predict_sentiment(input_text):
    # Transform input text using the trained TF-IDF vectorizer
    X_input_tfidf = tfidf_vectorizer.transform([input_text])

    # Predict sentiment using the trained logistic regression model
    sentiment_score = loaded_model.predict(X_input_tfidf)
    return sentiment_score[0]

def main():
    st.title("Sentiment Analysis App")

    # User input text
    user_input = st.text_area("Enter your text:")

    if st.button("Analyze"):
        # Preprocess input text
        preprocessed_text = preprocess_text(user_input)

        # Predict sentiment
        sentiment_score = predict_sentiment(preprocessed_text)

        # Display sentiment score
        st.write(f"Sentiment Score: {sentiment_score}")

if __name__ == "__main__":
    main()
