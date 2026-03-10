"""
Model Trainer Module
Handles training and saving of ML classification models.
"""

import os

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


def train_naive_bayes(X_train, y_train, alpha=1.0):
    """
    Train a Multinomial Naive Bayes classifier.

    Args:
        X_train: TF-IDF training features
        y_train: Training labels
        alpha: Smoothing parameter (default: 1.0)

    Returns:
        Trained MultinomialNB model
    """
    print("Training Naive Bayes classifier...")
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)
    print("  Naive Bayes training complete.")
    return model


def train_logistic_regression(X_train, y_train, max_iter=1000):
    """
    Train a Logistic Regression classifier.

    Args:
        X_train: TF-IDF training features
        y_train: Training labels
        max_iter: Maximum iterations (default: 1000)

    Returns:
        Trained LogisticRegression model
    """
    print("Training Logistic Regression classifier...")
    model = LogisticRegression(max_iter=max_iter, random_state=42, C=1.0)
    model.fit(X_train, y_train)
    print("  Logistic Regression training complete.")
    return model


def save_model(model, path):
    """
    Save a trained model to disk.

    Args:
        model: Trained model
        path: File path to save the model
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"  Model saved to {path}")


def load_model(path):
    """
    Load a trained model from disk.

    Args:
        path: File path of the saved model

    Returns:
        Loaded model
    """
    return joblib.load(path)


def train_all_models(X_train, y_train, save_dir="models"):
    """
    Train all models and save them.

    Args:
        X_train: TF-IDF training features
        y_train: Training labels
        save_dir: Directory to save models

    Returns:
        Dictionary of model name -> trained model
    """
    os.makedirs(save_dir, exist_ok=True)

    models = {}

    # Train Naive Bayes
    nb_model = train_naive_bayes(X_train, y_train)
    save_model(nb_model, os.path.join(save_dir, "naive_bayes.joblib"))
    models["Naive Bayes"] = nb_model

    # Train Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train)
    save_model(lr_model, os.path.join(save_dir, "logistic_regression.joblib"))
    models["Logistic Regression"] = lr_model

    print(f"\nAll models trained and saved to '{save_dir}/' directory.")
    return models
