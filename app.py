"""
Flask Web Application
Simple web interface for fake news prediction.
Usage: python app.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify
from src.predictor import predict, load_pipeline

app = Flask(__name__)

# Load model and vectorizer once at startup
model = None
vectorizer = None


def get_model():
    """Lazy-load the model and vectorizer."""
    global model, vectorizer
    if model is None or vectorizer is None:
        try:
            model, vectorizer = load_pipeline(
                model_path="models/logistic_regression.joblib",
                vectorizer_path="models/tfidf_vectorizer.joblib",
            )
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            print("Please run 'python train.py' first to train the models.")
            sys.exit(1)
    return model, vectorizer


@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_news():
    """Handle prediction requests."""
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Please enter some news text."}), 400

    if len(text) < 10:
        return jsonify({"error": "Please enter a longer news article (at least 10 characters)."}), 400

    m, v = get_model()
    result = predict(text, model=m, vectorizer=v)

    return jsonify({
        "prediction": result["prediction"],
        "confidence": round(result["confidence"], 2) if result["confidence"] else None,
        "label_code": result["label_code"],
    })


if __name__ == "__main__":
    print("\n  Loading model...")
    get_model()
    print("  Model loaded successfully!")
    print("\n  Starting Fake News Detection Web App...")
    print("  Open http://localhost:5000 in your browser\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
