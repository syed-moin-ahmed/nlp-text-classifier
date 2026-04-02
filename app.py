import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pdfplumber
import ssl
import time

# Fix SSL for nltk
try:
    _create_unverified_https_context = ssl._create_unverified_context
except:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download nltk data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Page config
st.set_page_config(page_title="Resume Classifier AI", layout="wide")

st.title("📄 AI Resume Classifier")

# ✅ LOAD MODELS (FIXED)
@st.cache_resource
def load_models():
    try:
        tfidf = joblib.load("model/tfidf_vectorizer.pkl")
        model = joblib.load("model/best_model.pkl")
        encoder = joblib.load("model/label_encoder.pkl")
        return tfidf, model, encoder
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

tfidf, model, encoder = load_models()

# ✅ TEXT PREPROCESSING
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return " ".join(tokens)

# ✅ PDF TEXT EXTRACTION
def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text += content
    except:
        st.error("Error reading PDF")
    return text

# INPUT
option = st.radio("Choose Input Method", ["Paste Text", "Upload PDF"])

resume_text = ""

if option == "Paste Text":
    resume_text = st.text_area("Enter Resume Text")

else:
    file = st.file_uploader("Upload PDF", type=["pdf"])
    if file:
        resume_text = extract_text_from_pdf(file)
        st.success("PDF Loaded")

# PREDICT
if st.button("Predict"):
    if resume_text.strip() == "":
        st.warning("Please enter text")
    else:
        cleaned = preprocess_text(resume_text)

        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)
        category = encoder.inverse_transform(prediction)

        st.success(f"Predicted Category: {category[0]}")
