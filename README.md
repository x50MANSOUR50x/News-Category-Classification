# 📰 News Category Classification (AG News)

This project is a **machine learning pipeline** for classifying news articles into one of four categories using models like **Logistic Regression, SVM, Random Forest, and XGBoost**, with both **TF-IDF** and **GloVe embeddings**.

---

## 📂 Dataset

- Dataset: [AG News Dataset on Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
- Classes:
  - 1: World
  - 2: Sports
  - 3: Business
  - 4: Sci/Tech

---

## 🚀 Features

- Preprocessing with NLTK (stopword removal, stemming)
- TF-IDF vectorization (unigrams + bigrams)
- GloVe word embeddings (100d)
- Multiple classifiers:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
  - XGBoost
- Evaluation Metrics:
  - Accuracy
  - Macro F1 Score
  - Confusion Matrix
- Visualizations:
  - Class distribution
  - Model confusion matrices
  - Most influential words per class

---

## 🔧 Setup

### 1️⃣ Clone the repo

git clone https://github.com/your-username/ag-news-classification.git
cd ag-news-classification

### 2️⃣ Install dependencies
pip install -r requirements.txt

### 3️⃣ Download NLTK stopwords
import nltk
nltk.download('stopwords')

### 4️⃣ (Optional) Download GloVe
Download glove.6B.100d.txt and place it in:

Data/glove.6B.100d.txt

### ▶️ Run the Notebook
jupyter notebook ag_news_classification.ipynb

📁 Project Structure
📦 ag-news-classification/

├── Data/
│   └── train.csv, test.csv, glove.6B.100d.txt
├── news_model.pkl
├── tfidf_vectorizer.pkl
├── app.py
├── ag_news_classification.ipynb
├── requirements.txt
├── .gitignore
└── README.md


🤝 Credits

Developed by Mohammed Ahmed Mansour

Internship Project – Elevvo