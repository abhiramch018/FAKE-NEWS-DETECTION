"""
Text Preprocessor Module
Handles all NLP preprocessing steps for news text data.
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def clean_text(text):
    """
    Clean a single text string by applying all preprocessing steps.

    Steps:
        1. Convert to lowercase
        2. Remove URLs
        3. Remove HTML tags
        4. Remove punctuation and special characters
        5. Remove numbers
        6. Tokenize
        7. Remove stopwords
        8. Apply stemming

    Args:
        text: Raw text string

    Returns:
        Cleaned and preprocessed text string
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove punctuation and special characters
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and apply stemming
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]

    return " ".join(tokens)


def preprocess_dataframe(df):
    """
    Apply preprocessing to the entire DataFrame.

    Args:
        df: DataFrame with 'title' and 'text' columns

    Returns:
        DataFrame with added 'cleaned_content' column
    """
    print("Preprocessing text data...")

    # Combine title and text
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

    # Apply cleaning
    df["cleaned_content"] = df["content"].apply(clean_text)

    # Show progress
    total = len(df)
    print(f"  Preprocessed {total} articles successfully.")

    return df


def preprocess_single(text):
    """
    Preprocess a single text input (for prediction).

    Args:
        text: Raw text string

    Returns:
        Cleaned text string
    """
    return clean_text(text)
