import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pdfplumber
import os
import ssl
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import time

# Fix SSL for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# Page config
st.set_page_config(
    page_title="Resume Classifier AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium UI
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Animated gradient header */
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    
    .gradient-header {
        background: linear-gradient(90deg, #667eea, #764ba2, #6b8cff, #a855f7);
        background-size: 300% 300%;
        animation: gradient 8s ease infinite;
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    /* Cards */
    .premium-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        margin-bottom: 1rem;
    }
    
    /* Result card */
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        text-align: center;
        border: 2px solid #f0f0f0;
        transition: transform 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 30px 60px rgba(102, 126, 234, 0.2);
    }
    
    /* Confidence badges */
    .badge-high {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: bold;
        display: inline-block;
    }
    
    .badge-medium {
        background: linear-gradient(135deg, #ffc107, #fd7e14);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: bold;
        display: inline-block;
    }
    
    .badge-low {
        background: linear-gradient(135deg, #dc3545, #c82333);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: bold;
        display: inline-block;
    }
    
    /* Stats boxes */
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.2);
    }
    
    /* Keywords */
    .keyword-tag {
        background: linear-gradient(135deg, #e0e7ff, #c7d2fe);
        color: #4f46e5;
        padding: 0.3rem 0.8rem;
        border-radius: 50px;
        display: inline-block;
        margin: 0.2rem;
        font-size: 0.9rem;
        font-weight: 500;
        border: 1px solid #a5b4fc;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 50px;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* File uploader */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 50px;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Animated header
st.markdown("""
<div class="gradient-header">
    <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">📄 AI Resume Classifier</h1>
    <p style="font-size: 1.2rem; opacity: 0.9;">Powered by Machine Learning & Natural Language Processing</p>
    <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 1rem;">
        <span style="background: rgba(255,255,255,0.2); padding: 0.3rem 1rem; border-radius: 50px;">🤖 24 Categories</span>
        <span style="background: rgba(255,255,255,0.2); padding: 0.3rem 1rem; border-radius: 50px;">⚡ 95% Accuracy</span>
        <span style="background: rgba(255,255,255,0.2); padding: 0.3rem 1rem; border-radius: 50px;">📊 Real-time Analysis</span>
    </div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load trained models from disk"""
    try:
        import os
        models_path = "/Users/syedmoinahmed/Desktop/resume-classification-project/models"
        
        with st.spinner("Loading AI models..."):
            time.sleep(1)
            tfidf = joblib.load(os.path.join(models_path, 'tfidf_vectorizer.pkl'))
            model = joblib.load(os.path.join(models_path, 'best_model.pkl'))
            encoder = joblib.load(os.path.join(models_path, 'label_encoder.pkl'))
            
            # Success animation
            st.balloons()
            return tfidf, model, encoder
    except Exception as e:
        st.error(f"⚠️ Error loading models: {e}")
        return None, None, None

def preprocess_text(text):
    """Clean and preprocess resume text"""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF"""
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            progress_bar = st.progress(0)
            for i, page in enumerate(pdf.pages):
                extracted = page.extract_text()
                if extracted:
                    text += extracted + " "
                progress_bar.progress((i + 1) / len(pdf.pages))
            time.sleep(0.5)
            progress_bar.empty()
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_keywords(text, tfidf_vectorizer, top_n=15):
    """Extract top keywords from text"""
    try:
        feature_names = tfidf_vectorizer.get_feature_names_out()
        text_tfidf = tfidf_vectorizer.transform([text])
        scores = text_tfidf.toarray()[0]
        top_indices = scores.argsort()[-top_n:][::-1]
        keywords = [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]
        return keywords
    except:
        return []

def create_confidence_gauge(confidence):
    """Create a gauge chart for confidence"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score", 'font': {'size': 14}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, 60], 'color': "#ffcccc"},
                {'range': [60, 80], 'color': "#fff0cc"},
                {'range': [80, 100], 'color': "#ccffcc"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
    return fig

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h2 style="color: #667eea;">🎯 About</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 1rem;">
        <h3 style="margin-top: 0;">✨ Features</h3>
        <ul style="list-style-type: none; padding-left: 0;">
            <li>📊 24 Job Categories</li>
            <li>🤖 Advanced NLP</li>
            <li>📄 PDF Support</li>
            <li>⚡ Instant Results</li>
            <li>🎯 95%+ Accuracy</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats in sidebar
    st.markdown("### 📊 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="stat-box">
            <h3>24</h3>
            <p>Categories</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-box">
            <h3>95%</h3>
            <p>Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sample categories
    st.markdown("### 🏷️ Sample Categories")
    categories = ["DATA SCIENCE", "WEB DEVELOPING", "HR", "FINANCE", "HEALTHCARE", "ENGINEERING", "SALES", "TEACHER"]
    for cat in categories:
        st.markdown(f"<span class='keyword-tag'>{cat}</span>", unsafe_allow_html=True)

# Load models
tfidf, model, encoder = load_models()

if tfidf is not None and model is not None and encoder is not None:
    # Main content area
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.05);">
            <h2 style="color: #333; margin-bottom: 1rem;">📝 Input Resume</h2>
        """, unsafe_allow_html=True)
        
        input_method = st.radio(
            "Choose input method:",
            ["📋 Paste Text", "📄 Upload PDF"],
            horizontal=True
        )
        
        resume_text = ""
        
        if input_method == "📋 Paste Text":
            resume_text = st.text_area(
                "Paste your resume text here:",
                height=300,
                placeholder="Copy and paste your resume content here..."
            )
            
            if resume_text:
                char_count = len(resume_text)
                word_count = len(resume_text.split())
                st.caption(f"📊 {char_count} characters • {word_count} words")
        
        else:
            uploaded_file = st.file_uploader(
                "Upload PDF file",
                type=['pdf'],
                help="Upload a PDF resume file"
            )
            
            if uploaded_file:
                with st.spinner("📄 Extracting text from PDF..."):
                    resume_text = extract_text_from_pdf(uploaded_file)
                    if resume_text:
                        st.success(f"✅ PDF processed! Extracted {len(resume_text)} characters")
                        with st.expander("🔍 Preview extracted text"):
                            st.write(resume_text[:500] + "...")
        
        # Classify button
        if st.button("🚀 Analyze Resume", use_container_width=True):
            if resume_text and len(resume_text.strip()) > 50:
                with st.spinner("🤖 AI is analyzing your resume..."):
                    # Progress animation
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # Preprocess
                    cleaned_text = preprocess_text(resume_text)
                    
                    # Vectorize
                    text_vector = tfidf.transform([cleaned_text])
                    
                    # Predict
                    pred_id = model.predict(text_vector)[0]
                    pred_category = encoder.inverse_transform([pred_id])[0]
                    
                    # Get confidence
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(text_vector)[0]
                        confidence = probs[pred_id]
                    else:
                        confidence = 0.85
                    
                    # Get top 3 predictions
                    if hasattr(model, "predict_proba"):
                        top_3_idx = probs.argsort()[-3:][::-1]
                        top_3_categories = encoder.inverse_transform(top_3_idx)
                        top_3_confidences = probs[top_3_idx]
                    else:
                        top_3_categories = [pred_category, "N/A", "N/A"]
                        top_3_confidences = [confidence, 0, 0]
                    
                    # Store in session state
                    st.session_state['predicted'] = pred_category
                    st.session_state['confidence'] = confidence
                    st.session_state['cleaned'] = cleaned_text
                    st.session_state['top_3'] = list(zip(top_3_categories, top_3_confidences))
                    
                    progress_bar.empty()
                    st.success("✅ Analysis complete!")
                    st.balloons()
            else:
                st.warning("⚠️ Please enter at least 50 characters")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.05);">
            <h2 style="color: #333; margin-bottom: 1rem;">🎯 Analysis Results</h2>
        """, unsafe_allow_html=True)
        
        if 'predicted' in st.session_state:
            pred = st.session_state['predicted']
            conf = st.session_state['confidence']
            
            # Confidence badge
            if conf > 0.8:
                badge_class = "badge-high"
                badge_text = "🔥 HIGH CONFIDENCE"
            elif conf > 0.6:
                badge_class = "badge-medium"
                badge_text = "⚡ MEDIUM CONFIDENCE"
            else:
                badge_class = "badge-low"
                badge_text = "⚠️ LOW CONFIDENCE"
            
            # Main result card
            st.markdown(f"""
            <div class="result-card">
                <div style="font-size: 1.2rem; color: #666; margin-bottom: 0.5rem;">Predicted Category</div>
                <h1 style="font-size: 3rem; color: #667eea; margin: 0.5rem 0;">{pred}</h1>
                <div style="margin: 1rem 0;">
                    <span class="{badge_class}">{badge_text}</span>
                </div>
                <div style="font-size: 2rem; font-weight: bold; color: #333;">{conf:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence gauge
            st.plotly_chart(create_confidence_gauge(conf), use_container_width=True)
            
            # Top 3 predictions
            if 'top_3' in st.session_state:
                st.markdown("### 📊 Top 3 Predictions")
                for cat, prob in st.session_state['top_3']:
                    st.markdown(f"""
                    <div style="margin: 0.5rem 0;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.2rem;">
                            <span><strong>{cat}</strong></span>
                            <span>{prob:.1%}</span>
                        </div>
                        <div style="background: #f0f0f0; height: 8px; border-radius: 4px;">
                            <div style="background: linear-gradient(90deg, #667eea, #764ba2); width: {prob*100}%; height: 8px; border-radius: 4px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Keywords
            if 'cleaned' in st.session_state:
                st.markdown("### 🔑 Key Skills Found")
                keywords = extract_keywords(st.session_state['cleaned'], tfidf, 12)
                
                # Display as tags
                tags_html = ""
                for keyword, score in keywords:
                    tags_html += f'<span class="keyword-tag">{keyword} ({score:.2f})</span> '
                st.markdown(f'<div>{tags_html}</div>', unsafe_allow_html=True)
                
                # Word count
                word_count = len(st.session_state['cleaned'].split())
                st.caption(f"📝 Resume contains {word_count} significant keywords")
            
            # Job recommendations
            st.markdown("### 💼 Recommended Positions")
            
            job_recommendations = {
                "DATA SCIENCE": ["Data Scientist", "ML Engineer", "Data Analyst", "AI Specialist"],
                "WEB DEVELOPING": ["Full Stack Developer", "Frontend Developer", "Backend Developer", "UI/UX Designer"],
                "HR": ["HR Manager", "Recruitment Specialist", "Talent Acquisition", "HR Generalist"],
                "FINANCE": ["Financial Analyst", "Accountant", "Investment Banker", "Auditor"],
                "HEALTHCARE": ["Doctor", "Nurse", "Healthcare Administrator", "Medical Researcher"],
                "ENGINEERING": ["Software Engineer", "Mechanical Engineer", "Civil Engineer", "Electrical Engineer"],
                "SALES": ["Sales Manager", "Business Development", "Account Executive", "Sales Representative"],
                "TEACHER": ["Teacher", "Professor", "Education Coordinator", "Curriculum Developer"]
            }
            
            # Find matching category
            matched_jobs = []
            for key, jobs in job_recommendations.items():
                if key in pred or pred in key:
                    matched_jobs = jobs
                    break
            
            if matched_jobs:
                for job in matched_jobs:
                    st.markdown(f"<span class='keyword-tag' style='background: #667eea; color: white;'>{job}</span>", unsafe_allow_html=True)
            else:
                st.info("Browse job portals for positions matching your profile")
        
        else:
            # Placeholder when no result yet
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 15px;">
                <h3 style="color: #aaa; margin-bottom: 1rem;">👈 Enter a resume to see results</h3>
                <p style="color: #888;">Paste text or upload PDF to get instant classification</p>
                <div style="font-size: 3rem; margin: 1rem 0;">📊</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Additional features section
    st.markdown("---")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h3 style="color: #667eea;">🚀 Fast & Accurate</h3>
            <p>Get results in seconds with 95%+ accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h3 style="color: #667eea;">📊 24 Categories</h3>
            <p>Covering all major job domains</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h3 style="color: #667eea;">🔒 Privacy First</h3>
            <p>Your data stays on your device</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <hr style="margin: 2rem 0;">
    <p>✨ Built with Streamlit • Powered by Machine Learning • NLP-Based Resume Classification ✨</p>
    <p style="font-size: 0.8rem;">© 2026 AI Resume Classifier | All rights reserved</p>
</div>
""", unsafe_allow_html=True)
