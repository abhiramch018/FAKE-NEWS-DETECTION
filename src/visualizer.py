"""
Visualizer Module
Creates charts and plots for model evaluation and analysis.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def setup_style():
    """Set up matplotlib style for consistent, attractive plots."""
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def plot_accuracy_comparison(all_metrics, save_path="outputs/accuracy_comparison.png"):
    """
    Create a bar chart comparing model accuracies.

    Args:
        all_metrics: Dictionary of model name -> metrics
        save_path: Path to save the plot
    """
    setup_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model_names = list(all_metrics.keys())
    accuracies = [all_metrics[name]["accuracy"] * 100 for name in model_names]

    # Define colors
    colors = ["#4E79A7", "#E15759", "#76B7B2", "#59A14F"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(model_names, accuracies, color=colors[:len(model_names)],
                  edgecolor="white", linewidth=1.5, width=0.5)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{acc:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=13)

    ax.set_title("Model Accuracy Comparison", fontweight="bold", fontsize=16, pad=15)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_ylim(0, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Accuracy comparison chart saved to {save_path}")


def plot_confusion_matrix(cm, model_name, save_path=None):
    """
    Plot a confusion matrix heatmap.

    Args:
        cm: Confusion matrix array
        model_name: Name of the model
        save_path: Path to save the plot (optional)
    """
    setup_style()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Fake", "Real"],
        yticklabels=["Fake", "Real"],
        ax=ax, linewidths=1, linecolor="white",
        annot_kws={"size": 16, "weight": "bold"},
    )
    ax.set_title(f"Confusion Matrix - {model_name}", fontweight="bold", fontsize=16, pad=15)
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Confusion matrix saved to {save_path}")
    else:
        plt.show()


def plot_all_confusion_matrices(all_metrics, save_dir="outputs"):
    """
    Plot confusion matrices for all models.

    Args:
        all_metrics: Dictionary of model name -> metrics
        save_dir: Directory to save plots
    """
    for name, metrics in all_metrics.items():
        safe_name = name.lower().replace(" ", "_")
        save_path = os.path.join(save_dir, f"confusion_matrix_{safe_name}.png")
        plot_confusion_matrix(metrics["confusion_matrix"], name, save_path)


def plot_metrics_comparison(all_metrics, save_path="outputs/metrics_comparison.png"):
    """
    Create a grouped bar chart comparing all metrics across models.

    Args:
        all_metrics: Dictionary of model name -> metrics
        save_path: Path to save the plot
    """
    setup_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model_names = list(all_metrics.keys())
    metric_names = ["accuracy", "precision", "recall", "f1_score"]
    display_names = ["Accuracy", "Precision", "Recall", "F1 Score"]

    x = np.arange(len(display_names))
    width = 0.25
    colors = ["#4E79A7", "#E15759", "#76B7B2", "#59A14F"]

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, name in enumerate(model_names):
        values = [all_metrics[name][m] * 100 for m in metric_names]
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=name,
                      color=colors[i % len(colors)], edgecolor="white", linewidth=1)
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("Model Performance Comparison", fontweight="bold", fontsize=16, pad=15)
    ax.set_ylabel("Score (%)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=12)
    ax.set_ylim(0, 108)
    ax.legend(fontsize=11, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Metrics comparison chart saved to {save_path}")


def plot_feature_importance(vectorizer, model, model_name,
                            top_n=20, save_path="outputs/feature_importance.png"):
    """
    Plot top TF-IDF feature importance for a model.

    Args:
        vectorizer: Fitted TF-IDF vectorizer
        model: Trained model (must have coef_ attribute for LR)
        model_name: Name of the model
        top_n: Number of top features to display
        save_path: Path to save the plot
    """
    setup_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    feature_names = vectorizer.get_feature_names_out()

    if hasattr(model, "coef_"):
        # For Logistic Regression
        importance = model.coef_[0]
    elif hasattr(model, "feature_log_prob_"):
        # For Naive Bayes - difference between class log probabilities
        importance = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    else:
        print(f"  Cannot extract feature importance for {model_name}")
        return

    # Get top positive and negative features
    top_positive_idx = np.argsort(importance)[-top_n:]
    top_negative_idx = np.argsort(importance)[:top_n]

    top_idx = np.concatenate([top_negative_idx, top_positive_idx])
    top_features = feature_names[top_idx]
    top_values = importance[top_idx]

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = ["#E15759" if v < 0 else "#4E79A7" for v in top_values]
    ax.barh(range(len(top_features)), top_values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=10)
    ax.set_title(f"Top {top_n} Features - {model_name}\n(Red = Fake indicators, Blue = Real indicators)",
                 fontweight="bold", fontsize=14, pad=15)
    ax.set_xlabel("Feature Importance", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Feature importance chart saved to {save_path}")


def plot_label_distribution(df, save_path="outputs/label_distribution.png"):
    """
    Plot the distribution of labels in the dataset.

    Args:
        df: DataFrame with 'label' column
        save_path: Path to save the plot
    """
    setup_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    counts = df["label"].value_counts()
    labels = ["Fake News", "Real News"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    colors = ["#E15759", "#4E79A7"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    bars = axes[0].bar(labels, values, color=colors, edgecolor="white", linewidth=1.5, width=0.5)
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                     f"{val:,}", ha="center", va="bottom", fontweight="bold", fontsize=13)
    axes[0].set_title("Label Distribution", fontweight="bold", fontsize=14)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Pie chart
    axes[1].pie(values, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=90, textprops={"fontsize": 12, "fontweight": "bold"},
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Label Proportion", fontweight="bold", fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Label distribution chart saved to {save_path}")
