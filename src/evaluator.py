"""
Model Evaluator Module
Evaluates trained models using standard classification metrics.
"""

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained model on test data.

    Args:
        model: Trained classification model
        X_test: Test features
        y_test: True test labels
        model_name: Name of the model for display

    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Fake", "Real"]
        ),
        "predictions": y_pred,
    }

    return metrics


def print_metrics(metrics):
    """
    Print evaluation metrics in a formatted way.

    Args:
        metrics: Dictionary of evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"  {metrics['model_name']} - Evaluation Results")
    print(f"{'='*60}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1_score']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  {metrics['confusion_matrix']}")
    print(f"\n  Classification Report:")
    print(metrics["classification_report"])
    print(f"{'='*60}\n")


def evaluate_all_models(models, X_test, y_test):
    """
    Evaluate all trained models.

    Args:
        models: Dictionary of model name -> trained model
        X_test: Test features
        y_test: True test labels

    Returns:
        Dictionary of model name -> evaluation metrics
    """
    all_metrics = {}

    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, model_name=name)
        print_metrics(metrics)
        all_metrics[name] = metrics

    return all_metrics


def get_best_model(all_metrics):
    """
    Identify the best model based on accuracy.

    Args:
        all_metrics: Dictionary of model name -> metrics

    Returns:
        Tuple of (best_model_name, best_accuracy)
    """
    best_name = max(all_metrics, key=lambda k: all_metrics[k]["accuracy"])
    best_acc = all_metrics[best_name]["accuracy"]
    print(f"Best Model: {best_name} with accuracy {best_acc:.4f} ({best_acc*100:.2f}%)")
    return best_name, best_acc
