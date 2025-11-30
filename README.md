# Sentiment Analysis on Amazon Reviews  
CS-470: Machine Learning – End Semester Project  
Bachelor of Electrical Engineering  
SEECS  Fall 2025  
National University of Sciences & Technology (NUST)  

## 📌 Project Overview
This project performs **binary sentiment classification** on Amazon product reviews  
(Positive vs Negative). It compares **classical machine learning (SVM)** with a  
**deep learning model (1D CNN)** to determine which approach performs better.

The project strictly follows the CS-470 course requirements.

---

## 👥 Team Members
- Student 1:  Muhammad Ubaid Ur Rehman
- Student 2:  Aaina Abrar

---

## 🔍 Abstract
This project analyzes Amazon customer reviews using machine learning and deep learning techniques. 
The goal is to classify reviews into positive or negative sentiment.  
We implemented two models:  
1. **Support Vector Machine (SVM)** using TF-IDF  
2. **Convolutional Neural Network (CNN)** using word embeddings  

The CNN achieved better performance, demonstrating its ability to capture contextual patterns 
in text data. The complete methodology, results, and comparison between classical ML and deep learning models are included.

---

## 📂 Dataset
- **Amazon Reviews** (train.csv & test.csv)
- Size: ~3M rows (downsampled optionally)
- Each row contains:
  - `reviewText`: the customer review  
  - `overall`: rating 1–5 (converted to binary sentiment)  
- Source: Amazon Review Dataset

---

## 🧹 Preprocessing Pipeline
The following preprocessing steps were applied:

### ✔ Text Cleaning
- Lowercasing  
- Removing punctuation  
- Removing numbers  
- Removing URLs  
- Removing extra spaces  

### ✔ Token Preprocessing
- Stopword removal  
- Lemmatization  
- Optional stemming  

### ✔ Label Engineering
- Rating 1–2 → Negative
- Rating 4–5 → Positive
- Rating 3 → Removed (neutral)


### ✔ Train/Validation Split
- 80% training  
- 20% validation  

---

## 🔎 Exploratory Data Analysis (EDA)
Performed:
- Class distribution plot  
- Wordcloud of most frequent positive & negative words  
- Review length distribution  
- Most common n-grams  

---

## 🧠 Classical Machine Learning Model: SVM
### **Vectorization**
- TF-IDF (max_features=50,000)

### **Model**
- Support Vector Machine (linear kernel)

### **Evaluation**
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix  

---

## 🤖 Deep Learning Model: 1D CNN
### Architecture
- Tokenizer + padding  
- Embedding layer  
- 1D Convolution  
- MaxPooling  
- Dense layers  
- Dropout regularization  

### Evaluation
- Training/validation accuracy/loss plots  
- Confusion matrix  
- Full metric report  

---

## 📊 Model Comparison

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| SVM   | auto generated | auto generated | auto generated | auto generated |
| CNN   | auto generated | auto generated | auto generated | auto generated |

*(Table is auto-generated in notebook)*

---

## 📝 Automatic Conclusion
Based on automatic evaluation, the notebook prints:

- **Which model performs better**  
- **Why it performed better**  
- **Future work directions**  

Typically CNN outperforms SVM due to its ability to learn contextual features at scale.

---

## 📁 Repository Structure  
📦 sentiment-analysis-amazon  
┣ 📂 data  
┃ ┣ train.csv  
┃ ┗ test.csv  
┣ 📂 models  
┃ ┣ svm_model.pkl  
┃ ┗ cnn_model.h5  
┣ 📂 results  
┃ ┣ svm_confusion_matrix.png  
┃ ┣ cnn_confusion_matrix.png  
┃ ┣ training_curve.png  
┃ ┗ comparison_table.csv  
┣ 📜 sentiment_analysis.ipynb  
┣ 📜 requirements.txt  
┗ 📜 README.md  


---

## 🧪 Requirements
numpy  
pandas  
matplotlib  
seaborn  
nltk  
scikit-learn  
tensorflow / keras  
wordcloud  


---

## 🎯 Conclusion
The CNN model outperforms the SVM classifier due to:
- ability to capture local context via convolution filters  
- dense embeddings  
- better generalization on long text sequences  

SVM is still competitive and faster to train, but CNN is superior for large text datasets.

---

## 🔮 Future Work
- Add pretrained embeddings (GloVe, FastText)  
- Use LSTM, GRU, or Transformer models  
- Try BERT for state-of-the-art performance  
- Implement hyperparameter tuning for CNN  

---

# End of README
