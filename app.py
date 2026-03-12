import streamlit as st
import joblib
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)

@st.cache_resource
def load_model():
    return joblib.load("models/spam_classifier_pipeline.pkl")

model = load_model()

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

st.title("Spam Message Detector")

msg = st.text_area("Enter message")

if st.button("Check"):

    if msg.strip() == "":
        st.info("Enter some text first.")
    else:
        processed = clean_text(msg)

        pred = model.predict([processed])[0]
        prob = model.predict_proba([processed])[0]
        conf = round(max(prob)*100, 2)

        if pred == 1:
            st.error(f"Spam ({conf}% confidence)")
        else:
            st.success(f"Not Spam ({conf}% confidence)")

        with st.expander("processed text"):
            st.write(processed) 
