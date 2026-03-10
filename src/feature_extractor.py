"""
Feature Extractor Module
Handles TF-IDF vectorization of text data.
"""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_vectorizer(max_features=5000):
    """
    Create a TF-IDF vectorizer.

    Args:
        max_features: Maximum number of features (default: 5000)

    Returns:
        TfidfVectorizer instance
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),     # Unigrams and bigrams
        min_df=2,               # Minimum document frequency
        max_df=0.95,            # Maximum document frequency
        sublinear_tf=True,      # Apply sublinear TF scaling
    )
    return vectorizer


def fit_transform(vectorizer, X_train):
    """
    Fit the vectorizer on training data and transform it.

    Args:
        vectorizer: TfidfVectorizer instance
        X_train: Training text data

    Returns:
        Sparse matrix of TF-IDF features
    """
    print("Fitting TF-IDF vectorizer on training data...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    print(f"  TF-IDF matrix shape: {X_train_tfidf.shape}")
    return X_train_tfidf


def transform(vectorizer, X_test):
    """
    Transform test data using a fitted vectorizer.

    Args:
        vectorizer: Fitted TfidfVectorizer instance
        X_test: Test text data

    Returns:
        Sparse matrix of TF-IDF features
    """
    X_test_tfidf = vectorizer.transform(X_test)
    return X_test_tfidf


def save_vectorizer(vectorizer, path="models/tfidf_vectorizer.joblib"):
    """Save the fitted vectorizer to disk."""
    joblib.dump(vectorizer, path)
    print(f"  Vectorizer saved to {path}")


def load_vectorizer(path="models/tfidf_vectorizer.joblib"):
    """Load a fitted vectorizer from disk."""
    return joblib.load(path)


def get_feature_names(vectorizer):
    """Get the feature names from a fitted vectorizer."""
    return vectorizer.get_feature_names_out()
