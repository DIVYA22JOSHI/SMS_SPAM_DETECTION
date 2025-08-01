📱 SMS Spam Detection using Machine Learning

This Streamlit web application detects whether an SMS message is **Spam or Not Spam** using Natural Language Processing (NLP) and a **Multinomial Naive Bayes** classifier. The app provides a user-friendly interface to test and visualize predictions in real-time.

---

### 🔍 Features

* **Text Preprocessing** with:

  * Lowercasing
  * Removing special characters
  * Stopword removal
  * Stemming using NLTK
* **Vectorization** using `CountVectorizer` (Bag of Words)
* **Model Training** with hyperparameter tuning (`alpha` value for Naive Bayes)
* **Performance Evaluation**:

  * Accuracy score
  * Confusion matrix heatmap
* **Real-time Prediction** of user-input messages

---

### 📁 Dataset

* The dataset used is the classic [SMSSpamCollection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset), which contains labeled SMS messages (`ham` or `spam`).
* It is read from a TSV file and preprocessed for training and prediction.

---

### ⚙️ Technologies Used

* Python
* Streamlit
* Pandas, NumPy
* NLTK (Natural Language Toolkit)
* Scikit-learn
* Matplotlib, Seaborn

---

### 🚀 How to Run the App

```bash
# 1. Clone the repository
git clone https://github.com/your-username/SMS_SPAM_DETECTION.git
cd SMS_SPAM_DETECTION

# 2. Install required libraries
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

### 🧠 Model Accuracy

* Automatically selects the best `alpha` (smoothing) parameter.
* Displays model accuracy and confusion matrix.
* Final model accuracy on test set: **\~95%** (depending on data split)

---

### 🗨️ Live Prediction Example

Type your own SMS in the input box to test whether it's spam or not.



