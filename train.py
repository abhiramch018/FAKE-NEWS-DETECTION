"""
Train Script
Main entry point for running the complete fake news detection pipeline.
Usage: python train.py
"""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_data, get_data_summary, split_data
from src.preprocessor import preprocess_dataframe
from src.feature_extractor import (
    create_tfidf_vectorizer,
    fit_transform,
    transform,
    save_vectorizer,
)
from src.model_trainer import train_all_models
from src.evaluator import evaluate_all_models, get_best_model
from src.visualizer import (
    plot_accuracy_comparison,
    plot_all_confusion_matrices,
    plot_metrics_comparison,
    plot_feature_importance,
    plot_label_distribution,
)


def main():
    print("\n" + "=" * 60)
    print("  FAKE NEWS DETECTION - Training Pipeline")
    print("=" * 60)

    start_time = time.time()

    # ─── Step 1: Data Loading ───────────────────────────────────
    print("\n[Step 1/6] Loading dataset...")
    try:
        df = load_data("data/Fake.csv", "data/True.csv")
    except FileNotFoundError:
        print("\n  ERROR: Dataset files not found!")
        print("  Please download the dataset from Kaggle:")
        print("  https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        print("  Place 'Fake.csv' and 'True.csv' in the 'data/' folder.")
        sys.exit(1)

    summary = get_data_summary(df)
    print(f"  Total articles : {summary['total_samples']:,}")
    print(f"  Fake articles  : {summary['fake_count']:,}")
    print(f"  Real articles  : {summary['real_count']:,}")
    print(f"  Avg text length: {summary['avg_text_length']:.0f} characters")

    # ─── Step 2: Data Preprocessing ────────────────────────────
    print("\n[Step 2/6] Preprocessing text data...")
    df = preprocess_dataframe(df)

    # Split data
    X_train_raw, X_test_raw, y_train, y_test = split_data(df)
    print(f"  Training set : {len(X_train_raw):,} articles")
    print(f"  Testing set  : {len(X_test_raw):,} articles")

    # Use cleaned content for TF-IDF
    # Map back cleaned content
    train_idx = X_train_raw.index
    test_idx = X_test_raw.index
    X_train_clean = df.loc[train_idx, "cleaned_content"]
    X_test_clean = df.loc[test_idx, "cleaned_content"]

    # ─── Step 3: Feature Extraction ────────────────────────────
    print("\n[Step 3/6] Extracting TF-IDF features...")
    vectorizer = create_tfidf_vectorizer(max_features=5000)
    X_train_tfidf = fit_transform(vectorizer, X_train_clean)
    X_test_tfidf = transform(vectorizer, X_test_clean)

    # Save vectorizer
    os.makedirs("models", exist_ok=True)
    save_vectorizer(vectorizer, "models/tfidf_vectorizer.joblib")

    # ─── Step 4: Model Training ────────────────────────────────
    print("\n[Step 4/6] Training models...")
    models = train_all_models(X_train_tfidf, y_train, save_dir="models")

    # ─── Step 5: Model Evaluation ──────────────────────────────
    print("\n[Step 5/6] Evaluating models...")
    all_metrics = evaluate_all_models(models, X_test_tfidf, y_test)
    best_name, best_acc = get_best_model(all_metrics)

    # ─── Step 6: Generating Visualizations ─────────────────────
    print("\n[Step 6/6] Generating visualizations...")
    os.makedirs("outputs", exist_ok=True)

    plot_label_distribution(df, "outputs/label_distribution.png")
    plot_accuracy_comparison(all_metrics, "outputs/accuracy_comparison.png")
    plot_all_confusion_matrices(all_metrics, "outputs")
    plot_metrics_comparison(all_metrics, "outputs/metrics_comparison.png")

    # Feature importance for Logistic Regression
    if "Logistic Regression" in models:
        plot_feature_importance(
            vectorizer, models["Logistic Regression"], "Logistic Regression",
            top_n=20, save_path="outputs/feature_importance_lr.png"
        )

    # Feature importance for Naive Bayes
    if "Naive Bayes" in models:
        plot_feature_importance(
            vectorizer, models["Naive Bayes"], "Naive Bayes",
            top_n=20, save_path="outputs/feature_importance_nb.png"
        )

    # ─── Done ──────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print(f"  Time elapsed  : {elapsed:.1f} seconds")
    print(f"  Best model    : {best_name} ({best_acc*100:.2f}%)")
    print(f"  Models saved  : models/")
    print(f"  Plots saved   : outputs/")
    print("=" * 60)
    print("\n  To make predictions, run:")
    print("    python -m src.predictor")
    print("  Or launch the web app:")
    print("    python app.py")
    print()


if __name__ == "__main__":
    main()
