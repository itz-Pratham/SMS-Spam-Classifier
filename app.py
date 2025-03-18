import streamlit as st
import numpy as np
import pickle
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load pre-trained model and TF-IDF vectorizer
with open('./Models/spam_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('./Models/Vectorizer.pkl', 'rb') as vec_file:
    tfidf = pickle.load(vec_file)

# Initialize NLP tools
ps = PorterStemmer()
stopwords_set = set(stopwords.words('english'))

# Text preprocessing function (same as training)
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    words = nltk.word_tokenize(text)  # Tokenization
    words = [word for word in words if word.isalpha()]  # Remove non-alphabetic words
    words = [word for word in words if word not in stopwords_set]  # Remove stopwords
    words = [ps.stem(word) for word in words]  # Stemming
    return ' '.join(words)

# Streamlit App Interface
st.title("📩 SMS Spam Classifier")
st.markdown("Enter a message below, and the model will predict whether it's **Spam or Ham**.")

# User Input
user_input = st.text_area("Enter your SMS message:", "")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message before classifying.")
    else:
        # Preprocess the input
        transformed_input = transform_text(user_input)

        # Convert to TF-IDF format
        input_vector = tfidf.transform([transformed_input]).toarray()

        # Make prediction
        prediction = model.predict(input_vector)

        # Display result
        if prediction[0] == 1:
            st.error("🚨 This message is **SPAM**!")
        else:
            st.success("✅ This message is **HAM** (Not Spam).")

st.sidebar.header("About")
st.sidebar.info("This is a simple SMS Spam Classifier using **NLP and Machine Learning**. The model is trained on SMS data and predicts whether a message is spam or ham.")

st.sidebar.markdown("**Built with:**")
st.sidebar.markdown("- 🐍 Python")
st.sidebar.markdown("- 🏆 Scikit-learn")
st.sidebar.markdown("- 📊 Streamlit")
