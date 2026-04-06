<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/NLTK-NLP-4EA94B?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge" />
</p>

<h1 align="center">🔍 Fake News Detection using NLP & Machine Learning</h1>

<p align="center">
  <em>An end-to-end machine learning pipeline that classifies news articles as <strong>Real</strong> or <strong>Fake</strong> using Natural Language Processing, achieving <strong>99.27% accuracy</strong> with Logistic Regression.</em>
</p>

---

## 📌 Overview

Fake news spreads rapidly and can mislead millions. This project builds a **machine learning-based system** to automatically classify news articles as **Real** or **Fake** using **Natural Language Processing (NLP)**.

The system processes raw text, extracts meaningful features using TF-IDF vectorization, and applies classification algorithms to detect misinformation — all wrapped in a modern **Flask web application** for real-time predictions.

### 🎯 Key Objectives

- ✅ Classify news articles into **Real** or **Fake** categories
- ✅ Build a complete **end-to-end NLP pipeline** (preprocessing → feature extraction → training → evaluation)
- ✅ Compare performance of **Multinomial Naive Bayes** vs **Logistic Regression**
- ✅ Evaluate results using **Accuracy, Precision, Recall, F1 Score & Confusion Matrix**
- ✅ Deploy a **Flask web app** for real-time fake news detection

---

## 🧠 Methodology

```
Raw Text → Preprocessing → TF-IDF Vectorization → ML Classification → Prediction
```

### 1️⃣ Data Preprocessing
| Step | Description |
|------|-------------|
| Lowercasing | Normalize all text to lowercase |
| URL & HTML Removal | Strip out web links and HTML tags |
| Punctuation Removal | Remove special characters |
| Tokenization | Split text into tokens using **NLTK** |
| Stopword Removal | Filter out common English stopwords |
| Stemming | Reduce words to root form using **Porter Stemmer** |

### 2️⃣ Feature Extraction
- **TF-IDF Vectorization** with **5,000 features** and **bigram support**
- Captures word importance relative to the entire corpus

### 3️⃣ Model Training
- **Multinomial Naive Bayes** — Probabilistic classifier ideal for text
- **Logistic Regression** — Linear model with high interpretability

### 4️⃣ Evaluation
- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix visualization
- Feature importance analysis

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Source** | [Kaggle — Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) |
| **Files** | `Fake.csv`, `True.csv` |
| **Total Samples** | **44,898** articles |
| **Fake Articles** | 23,481 |
| **Real Articles** | 21,417 |
| **Train/Test Split** | 80% / 20% |

---

## 📈 Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|:--------:|:---------:|:------:|:--------:|
| Naive Bayes | 95.51% | 95.52% | 95.51% | 95.51% |
| **Logistic Regression** | **99.27%** | **99.27%** | **99.27%** | **99.27%** |

> ✅ **Best Model:** Logistic Regression with **99.27% accuracy**

### 📊 Performance Visualization

<p align="center">
  <img src="https://raw.githubusercontent.com/abhiramch018/FAKE-NEWS-DETECTION/main/outputs/model_performance.png" alt="Model Performance Comparison" width="700"/>
</p>

### 🔢 Confusion Matrix — Logistic Regression

<p align="center">
  <img src="https://raw.githubusercontent.com/abhiramch018/FAKE-NEWS-DETECTION/main/outputs/confusion_matrix.png" alt="Confusion Matrix" width="500"/>
</p>

### 📉 Additional Visualizations

<details>
<summary>Click to expand all visualizations</summary>
<br>

**Accuracy Comparison**
<p align="center">
  <img src="https://raw.githubusercontent.com/abhiramch018/FAKE-NEWS-DETECTION/main/outputs/accuracy_comparison.png" alt="Accuracy Comparison" width="600"/>
</p>

**Metrics Comparison**
<p align="center">
  <img src="https://raw.githubusercontent.com/abhiramch018/FAKE-NEWS-DETECTION/main/outputs/metrics_comparison.png" alt="Metrics Comparison" width="600"/>
</p>

**Label Distribution**
<p align="center">
  <img src="https://raw.githubusercontent.com/abhiramch018/FAKE-NEWS-DETECTION/main/outputs/label_distribution.png" alt="Label Distribution" width="700"/>
</p>

**Feature Importance — Logistic Regression**
<p align="center">
  <img src="https://raw.githubusercontent.com/abhiramch018/FAKE-NEWS-DETECTION/main/outputs/feature_importance_lr.png" alt="Feature Importance LR" width="600"/>
</p>

**Feature Importance — Naive Bayes**
<p align="center">
  <img src="https://raw.githubusercontent.com/abhiramch018/FAKE-NEWS-DETECTION/main/outputs/feature_importance_nb.png" alt="Feature Importance NB" width="600"/>
</p>

**Confusion Matrix — Naive Bayes**
<p align="center">
  <img src="https://raw.githubusercontent.com/abhiramch018/FAKE-NEWS-DETECTION/main/outputs/confusion_matrix_naive_bayes.png" alt="Confusion Matrix NB" width="500"/>
</p>

</details>

---

## 🌐 Web Application

A modern **Flask-based web interface** for real-time fake news detection with a sleek dark theme UI.

<p align="center">
  <img src="https://raw.githubusercontent.com/abhiramch018/FAKE-NEWS-DETECTION/main/outputs/web_app_real.png" alt="Web App - Real News Detection" width="800"/>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/abhiramch018/FAKE-NEWS-DETECTION/main/outputs/web_app_fake.png" alt="Web App - Fake News Detection" width="800"/>
</p>

**Features:**
- 🎨 Modern dark-themed glassmorphism UI
- ⚡ Real-time article analysis with confidence scores
- 📋 Sample articles for quick testing
- 📱 Responsive two-column layout

---

## 🛠️ Technologies Used

| Technology | Purpose |
|:----------:|---------|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) | Core programming language |
| ![NLTK](https://img.shields.io/badge/NLTK-4EA94B?style=flat-square) | Text preprocessing & tokenization |
| ![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) | ML models & TF-IDF vectorization |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) | Data manipulation & analysis |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) | Numerical computing |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square) | Data visualization |
| ![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat-square) | Statistical visualization |
| ![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white) | Web application framework |

---

## 📂 Project Structure

```
FAKENEWSDETECTION/
│
├── 📁 data/                    # Dataset files (Fake.csv, True.csv)
│
├── 📁 models/                  # Saved trained models & vectorizer
│   ├── logistic_regression.joblib
│   ├── naive_bayes.joblib
│   └── tfidf_vectorizer.joblib
│
├── 📁 outputs/                 # Generated visualizations
│   ├── model_performance.png
│   ├── confusion_matrix.png
│   ├── accuracy_comparison.png
│   ├── metrics_comparison.png
│   ├── label_distribution.png
│   ├── feature_importance_lr.png
│   └── feature_importance_nb.png
│
├── 📁 src/                     # Source modules
│   ├── __init__.py
│   ├── data_loader.py          # Data loading & splitting
│   ├── preprocessor.py         # Text cleaning & preprocessing
│   ├── feature_extractor.py    # TF-IDF feature extraction
│   ├── model_trainer.py        # Model training & saving
│   ├── evaluator.py            # Model evaluation & metrics
│   ├── predictor.py            # Prediction interface
│   └── visualizer.py           # Chart & plot generation
│
├── 📁 templates/               # Flask HTML templates
│   └── index.html
│
├── app.py                      # Flask web application
├── train.py                    # Training pipeline script
├── requirements.txt            # Python dependencies
└── README.md
```

---

## ▶️ How to Run

### Prerequisites
- Python 3.10+
- pip

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/FAKE-NEWS-DETECTION.git
cd FAKE-NEWS-DETECTION
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
Download from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) and place `Fake.csv` and `True.csv` in the `data/` folder.

### 4. Train the Models
```bash
python train.py
```
This runs the full pipeline: data loading → preprocessing → TF-IDF → training → evaluation → visualization.

### 5. Launch the Web App
```bash
python app.py
```
Open `http://localhost:5000` in your browser to start detecting fake news!

---

## 🔮 Future Enhancements

- 🤖 Implement **BERT / Transformer-based** deep learning models
- 📱 Extend to **social media** fake news detection (tweets, posts)
- ☁️ Deploy as a **cloud-based application** (AWS / GCP / Azure)
- 🔗 Add **real-time news API** integration for live detection
- 📊 Add **LSTM / RNN** sequence models for comparison

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  <strong>⭐ If you found this project useful, please consider giving it a star!</strong>
</p>
