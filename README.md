# 🔍 Fake News Detection using NLP and Machine Learning

A machine learning system that classifies news articles as **Real** or **Fake** using Natural Language Processing techniques. Built with Python, scikit-learn, NLTK, and Flask.

---

## 📋 Features

- **NLP Preprocessing** – Tokenization, stopword removal, stemming, and text cleaning
- **TF-IDF Vectorization** – Convert text to numerical features with bigram support
- **ML Models** – Multinomial Naive Bayes and Logistic Regression
- **Evaluation Metrics** – Accuracy, Precision, Recall, F1 Score, Confusion Matrix
- **Visualizations** – Accuracy charts, confusion matrices, feature importance plots
- **Web Interface** – Modern dark-themed Flask app for real-time predictions
- **Jupyter Notebook** – Complete end-to-end analysis and demonstration

---

## 🗂️ Project Structure

```
FAKENWESDETECTION/
├── data/                         # Dataset (Fake.csv, True.csv)
├── models/                       # Saved trained models
├── outputs/                      # Generated plots and reports
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Data loading and splitting
│   ├── preprocessor.py           # NLP text preprocessing
│   ├── feature_extractor.py      # TF-IDF vectorization
│   ├── model_trainer.py          # Model training (NB + LR)
│   ├── evaluator.py              # Model evaluation metrics
│   ├── visualizer.py             # Chart generation
│   └── predictor.py              # Prediction on new text
├── templates/
│   └── index.html                # Web UI template
├── app.py                        # Flask web application
├── train.py                      # CLI training pipeline
├── fake_news_detection.ipynb     # Jupyter notebook
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

Download the **Fake and Real News Dataset** from Kaggle:
- [https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

Place `Fake.csv` and `True.csv` in the `data/` folder.

### 3. Train Models

```bash
python train.py
```

This will:
- Load and preprocess the dataset
- Extract TF-IDF features
- Train Naive Bayes and Logistic Regression models
- Evaluate models and print metrics
- Generate visualization plots in `outputs/`
- Save trained models to `models/`

### 4. Run Web App

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser to use the prediction interface.

### 5. Jupyter Notebook

```bash
jupyter notebook fake_news_detection.ipynb
```

---

## 🧪 Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming language |
| NLTK | Text preprocessing (tokenization, stemming, stopwords) |
| Scikit-learn | ML models, TF-IDF, evaluation metrics |
| Pandas / NumPy | Data handling and manipulation |
| Matplotlib / Seaborn | Visualization |
| Flask | Web application framework |
| Jupyter Notebook | Interactive development |

---

## 📊 ML Algorithms

- **Multinomial Naive Bayes** – Probabilistic classifier based on Bayes' theorem
- **Logistic Regression** – Linear model for binary classification

### NLP Pipeline

1. **Lowercase** conversion
2. **URL & HTML** removal
3. **Punctuation** removal
4. **Tokenization** (NLTK word_tokenize)
5. **Stopword** removal
6. **Stemming** (Porter Stemmer)
7. **TF-IDF Vectorization** (5000 features, bigrams)

---

## 📈 Expected Results

- Model accuracy: **~90%+**
- Generated visualizations include:
  - Model accuracy comparison chart
  - Confusion matrices
  - Feature importance analysis
  - Label distribution charts

---

## 🔮 Future Directions

- Use Deep Learning models (BERT, LSTM)
- Detect fake news from social media posts
- Deploy as a cloud-based real-time application
