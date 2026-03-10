"""
Data Loader Module
Handles loading, combining, and splitting the fake news dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(fake_path="data/Fake.csv", true_path="data/True.csv"):
    """
    Load fake and true news datasets, combine them with labels.

    Args:
        fake_path: Path to the Fake news CSV file
        true_path: Path to the True news CSV file

    Returns:
        Combined DataFrame with a 'label' column (0=Fake, 1=Real)
    """
    # Load datasets
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    # Add labels
    fake_df["label"] = 0  # Fake
    true_df["label"] = 1  # Real

    # Combine datasets
    df = pd.concat([fake_df, true_df], axis=0, ignore_index=True)

    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def get_data_summary(df):
    """
    Get a summary of the dataset.

    Args:
        df: DataFrame with news data

    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "total_samples": len(df),
        "fake_count": len(df[df["label"] == 0]),
        "real_count": len(df[df["label"] == 1]),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "avg_text_length": df["text"].str.len().mean(),
    }
    return summary


def split_data(df, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.

    Args:
        df: DataFrame with news data
        test_size: Proportion of data for testing (default: 0.2)
        random_state: Random seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    # Combine title and text for feature
    df["content"] = df["title"] + " " + df["text"]

    X = df["content"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test
