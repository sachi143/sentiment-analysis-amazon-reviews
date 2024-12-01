import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(filepath, text_column='reviewText', rating_column='overall'):
    """Load the Amazon product reviews dataset."""
    data = pd.read_csv(filepath)
    data[text_column] = data[text_column].replace(np.nan, '')
    data['sentiment'] = data[rating_column].apply(lambda x: 'positive' if x > 3 else 'negative' if x < 3 else 'neutral')
    return data

def clean_text(text):
    """Clean the text data by removing non-alphabetic characters and converting to lowercase."""
    return re.sub(r'[^a-zA-Z\s]', '', text).lower()

def preprocess_data(data, text_column='reviewText'):
    """Apply text cleaning to the review text column."""
    data[text_column] = data[text_column].apply(clean_text)
    return data

def vectorize_data(data, text_column='reviewText'):
    """Convert text data to numerical feature vectors using TF-IDF."""
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data[text_column])
    return X, vectorizer

def split_data(data, X, label_column='sentiment', test_size=0.2):
    """Split the data into training and testing sets."""
    y = data[label_column]
    return train_test_split(X, y, test_size=test_size, random_state=42)
