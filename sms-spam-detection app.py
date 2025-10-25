'''# sms_spam_classifier.py

import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK data (only first time)
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# ----------------------
# Text preprocessing
# ----------------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():  # keep only alphanumeric
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(ps.stem(i))  # apply stemming

    return " ".join(y)


# ----------------------
# TRAINING (only run once to create model.pkl & vectorizer.pkl)
# ----------------------
def train_and_save_model():
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB

    # Load dataset (replace with your spam.csv path)
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']]  # v1 = label, v2 = message
    df = df.rename(columns={'v1': 'label', 'v2': 'message'})

    df['message'] = df['message'].apply(transform_text)

    X = df['message']
    y = df['label']

    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)

    model = MultinomialNB()
    model.fit(X_vec, y)

    # Save model and vectorizer
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    st.success("Model trained and saved successfully!")


# ----------------------
# Load trained model & vectorizer
# ----------------------
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# ----------------------
# Streamlit app
# ----------------------
st.title("SMS Spam Classifier 📩")

menu = ["Predict SMS", "Train Model"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Predict SMS":
    st.subheader("Predict whether a message is Spam or Ham")
    user_input = st.text_area("Enter your message:")

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter a message to predict.")
        else:
            transformed_text = transform_text(user_input)
            vector_input = vectorizer.transform([transformed_text])
            result = model.predict(vector_input)[0]
            st.success(f"Prediction: {result}")

elif choice == "Train Model":
    st.subheader("Train the SMS Spam Classifier")
    st.info("This will retrain the model on your spam.csv dataset and overwrite existing model.pkl & vectorizer.pkl")
    if st.button("Train Now"):
        train_and_save_model()'''

# sms_spam_deployment_ready.py

import streamlit as st
import os
import string
import pickle
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# ----------------------
# NLTK setup
# ----------------------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
ps = PorterStemmer()

# ----------------------
# Text preprocessing
# ----------------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(ps.stem(i))

    return " ".join(y)

# ----------------------
# Load or train model
# ----------------------
MODEL_FILE = 'model.pkl'
VECTORIZER_FILE = 'vectorizer.pkl'

if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
    # Load pre-trained model
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_FILE, 'rb') as f:
        vectorizer = pickle.load(f)
else:
    st.info("Training model, please wait...")

    # Load dataset (make sure spam.csv is in the repo)
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']].rename(columns={'v1':'label', 'v2':'message'})

    df['message'] = df['message'].apply(transform_text)

    X = df['message']
    y = df['label']

    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)

    model = MultinomialNB()
    model.fit(X_vec, y)

    # Save for future use
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    with open(VECTORIZER_FILE, 'wb') as f:
        pickle.dump(vectorizer, f)

    st.success("Model trained successfully!")

# ----------------------
# Streamlit app UI
# ----------------------
st.title("SMS Spam Classifier 📩")

menu = ["Predict SMS", "About"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Predict SMS":
    st.subheader("Enter your message to predict Spam or Ham")
    user_input = st.text_area("Type your message here:")

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter a message to predict.")
        else:
            transformed_text = transform_text(user_input)
            vector_input = vectorizer.transform([transformed_text])
            result = model.predict(vector_input)[0]
            st.success(f"Prediction: {result}")

elif choice == "About":
    st.subheader("About this App")
    st.markdown("""
    - This is an SMS Spam Classifier built with Streamlit.
    - Uses **Naive Bayes** with **Bag-of-Words**.
    - Automatically trains the model if not found.
    - Preprocessing includes lowercase conversion, stopword removal, and stemming.
    """)

