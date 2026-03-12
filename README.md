# Spam Email Classifier

This project demonstrates a simple Natural Language Processing (NLP) and Machine Learning pipeline for classifying SMS or email messages as **Spam** or **Not Spam**.

It includes data preprocessing, model training and comparison, evaluation, and deployment using a Streamlit web application.

---

## Project Structure

```
spam-email-classifier/
│
├── data/
│   └── spam.csv
├── notebooks/
│   └── spam_classifier.ipynb
├── models/
│   └── spam_classifier_pipeline.pkl
├── app.py
├── requirements.txt
└── README.md
```

---

## Dataset

The model is trained using the **UCI SMS Spam Collection dataset** (available on Kaggle).

* Approximately 5,500 messages
* Two classes: `ham` (not spam) and `spam`
* The dataset is slightly imbalanced with more ham messages

---

## Text Preprocessing

The following preprocessing steps are applied:

* Conversion of text to lowercase
* Removal of punctuation
* Stopword removal using NLTK
* Stemming using Porter Stemmer
* Feature extraction using CountVectorizer and TF-IDF

---

## Models Evaluated

Multiple machine learning models are trained and compared:

* Multinomial Naive Bayes
* Logistic Regression
* Linear Support Vector Machine

Each model is evaluated using different vectorization techniques.
Based on evaluation metrics, **Logistic Regression with TF-IDF** is selected as the final model and saved as a pipeline.

---

## Running the Project

Clone the repository:

```bash
git clone https://github.com/Anusmita12/spam-email-classifier.git
cd spam-email-classifier
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the dataset (`spam.csv`) inside the `data/` directory.

Run the notebook to train and save the model.

Launch the Streamlit application:

```bash
streamlit run app.py
```

---

## Streamlit Application

The web application allows users to:

* Enter an SMS or email message
* Receive an instant spam prediction
* View the model’s confidence score

---

## Technologies Used

* pandas
* numpy
* matplotlib
* seaborn
* NLTK
* scikit-learn
* Streamlit
* joblib

---

## Author

**Anusmita**
GitHub: https://github.com/Anusmita12
