# Sentiment Analysis: Negative Movie Reviews

## Overview
This project was built for the **Film Junky Union**, a community of classic movie lovers aiming to automate the classification of user-submitted movie reviews. The objective was to build a robust sentiment analysis model capable of detecting **negative reviews**, achieving an F1 score of at least **0.85** on test data.

We used a dataset of IMDB movie reviews labeled by sentiment (positive/negative) and evaluated multiple models to determine the most accurate and efficient classifier.

---

## Problem Statement
Automatically detect **negative** movie reviews from user-generated text using machine learning models. The system must be both accurate and efficient, with a strong balance between precision and recall to handle class imbalance.

---

## Dataset
- **File**: `imdb_reviews.tsv`
- **Fields**:
  - `review`: The movie review text
  - `pos`: Sentiment label (0 = negative, 1 = positive)
  - `ds_part`: Indicates whether the record belongs to the training or testing dataset

*Dataset Source:*  
Andrew L. Maas et al. (2011). *Learning Word Vectors for Sentiment Analysis*. ACL 2011.

---


## Models Evaluated

### 1. **Dummy Classifier (Baseline)**
- **Accuracy**: ~50%
- **F1 Score**: ~0.33
- ➤ Performs no better than random guessing.

### 2. **Logistic Regression + TF-IDF (NLTK)**
- **Train Accuracy**: 94%
- **Test Accuracy**: 88%
- **F1 Score (Test)**: 0.88
- **ROC-AUC (Test)**: 0.95
- ➤ High performance and generalization, exceeded required F1 threshold.

### 3. **Logistic Regression + TF-IDF (spaCy)**
- Similar performance to NLTK version, with slightly different preprocessing pipeline.
- ➤ Also exceeded required performance thresholds.

### 4. **LGBMClassifier + TF-IDF (spaCy)**
- **Train Accuracy**: 91%
- **Test Accuracy**: 86%
- **F1 Score (Test)**: 0.86
- **ROC-AUC (Test)**: 0.94
- ➤ Strong performance, though slightly less generalization than Logistic Regression.

---

## Technologies Used
- Programming Languages
      - Python – Primary language for data processing, model development, and evaluation

- Data Handling & Preprocessing
      - Pandas – Data loading and manipulation
      - NumPy – Numerical computations
      - NLTK – Text preprocessing: tokenization, stopword removal, lemmatization
      - spaCy – Alternative NLP pipeline for tokenization and lemmatization
      - re (Regex) – Text cleaning and pattern matching

- Feature Engineering
     - Scikit-learn – TF-IDF vectorization, model training, evaluation metrics
     - scikit-learn’s Pipeline & GridSearch – Model tuning and reproducibility

- Machine Learning Models
     - Logistic Regression – Baseline classifier
     - LightGBM (LGBMClassifier) – Gradient boosting framework for performance testing
     - DummyClassifier – Baseline for comparison
     - BERT – Transformer-based embeddings on a small subset for experimentation

- Evaluation & Visualization
    - Matplotlib & Seaborn – Visualizing EDA results and model performance
    - Scikit-learn metrics – Accuracy, F1 Score, ROC-AUC, Average Precision Score

- Environment & Tools
    - Jupyter Notebook – Interactive code development and results presentation
    - joblib – Saving/loading trained models
    - TQDM – Progress bars for loops



