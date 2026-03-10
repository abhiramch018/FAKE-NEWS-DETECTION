"""
Predictor Module
Handles loading saved models and making predictions on new text.
"""

import os

import joblib

from src.preprocessor import preprocess_single


def load_pipeline(model_path="models/logistic_regression.joblib",
                  vectorizer_path="models/tfidf_vectorizer.joblib"):
    """
    Load a saved model and vectorizer for prediction.

    Args:
        model_path: Path to the saved model
        vectorizer_path: Path to the saved TF-IDF vectorizer

    Returns:
        Tuple of (model, vectorizer)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Run train.py first.")
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}. Run train.py first.")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def predict(text, model=None, vectorizer=None,
            model_path="models/logistic_regression.joblib",
            vectorizer_path="models/tfidf_vectorizer.joblib"):
    """
    Predict whether a news article is real or fake.

    Args:
        text: Raw news article text
        model: Pre-loaded model (optional)
        vectorizer: Pre-loaded vectorizer (optional)
        model_path: Path to saved model (used if model not provided)
        vectorizer_path: Path to saved vectorizer (used if vectorizer not provided)

    Returns:
        Dictionary with prediction result and confidence
    """
    # Load model and vectorizer if not provided
    if model is None or vectorizer is None:
        model, vectorizer = load_pipeline(model_path, vectorizer_path)

    # Preprocess the text
    cleaned_text = preprocess_single(text)

    # Vectorize
    text_tfidf = vectorizer.transform([cleaned_text])

    # Predict
    prediction = model.predict(text_tfidf)[0]
    label = "Real" if prediction == 1 else "Fake"

    # Get prediction probabilities if available
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(text_tfidf)[0]
        confidence = max(proba) * 100

    result = {
        "prediction": label,
        "label_code": int(prediction),
        "confidence": confidence,
        "cleaned_text_preview": cleaned_text[:200] + "..." if len(cleaned_text) > 200 else cleaned_text,
    }

    return result


def predict_batch(texts, model=None, vectorizer=None,
                  model_path="models/logistic_regression.joblib",
                  vectorizer_path="models/tfidf_vectorizer.joblib"):
    """
    Predict multiple news articles at once.

    Args:
        texts: List of raw news article texts
        model: Pre-loaded model (optional)
        vectorizer: Pre-loaded vectorizer (optional)

    Returns:
        List of prediction dictionaries
    """
    if model is None or vectorizer is None:
        model, vectorizer = load_pipeline(model_path, vectorizer_path)

    results = []
    for text in texts:
        result = predict(text, model=model, vectorizer=vectorizer)
        results.append(result)

    return results


# Interactive prediction mode
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Fake News Detection - Interactive Predictor")
    print("=" * 60)

    model, vectorizer = load_pipeline()

    while True:
        print("\nEnter a news article (or 'quit' to exit):")
        text = input("> ").strip()

        if text.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not text:
            print("Please enter some text.")
            continue

        result = predict(text, model=model, vectorizer=vectorizer)
        print(f"\n  Prediction : {result['prediction']}")
        if result["confidence"]:
            print(f"  Confidence : {result['confidence']:.2f}%")
