import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
import re
import warnings
warnings.filterwarnings('ignore')

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="AutoJudge",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
def load_css():
    st.markdown("""
    <style>
    /* Main styling */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    .header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .header h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
    }
    
    /* Card styling */
    .card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        border: 1px solid #e0e0e0;
    }
    
    .card-title {
        color: #333333;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Results styling */
    .result-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 2rem;
    }
    
    .result-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 0.5rem 0;
    }
    
    .result-label {
        font-size: 1rem;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 14px rgba(50, 50, 93, 0.1), 0 3px 6px rgba(0, 0, 0, 0.08);
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        transition: border-color 0.3s;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Status indicators */
    .status-easy {
        color: #2ecc71;
        font-weight: bold;
    }
    
    .status-medium {
        color: #f39c12;
        font-weight: bold;
    }
    
    .status-hard {
        color: #e74c3c;
        font-weight: bold;
    }
    
    /* Loading animation */
    .loading {
        text-align: center;
        padding: 2rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #888888;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Load custom CSS
load_css()

# Model loading with caching
@st.cache_resource
def load_models():
    """Load all ML models with caching"""
    try:
        models = {
            'clf': joblib.load("models/classification_model.pkl"),
            'reg': joblib.load("models/regression_model.pkl"),
            'vectorizer': joblib.load("models/vectorizer.pkl"),
            'scaler': joblib.load("models/scaler.pkl"),
            'label_encoder': joblib.load("models/label_encoder.pkl"),
            'svd': joblib.load("models/svd.pkl")
        }
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

# Text processing functions
def clean_text(text):
    """Clean and preprocess text"""
    if not text:
        return ""
    
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\t", " ", text)
    text = re.sub(r"[^a-z0-9+\-*/<>=(){}\[\],.: ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_complexity_features(text):
    """Extract programming complexity features from text"""
    text_lower = text.lower()
    
    features = [
        len(text),                                  # Total characters
        len(text.split()),                          # Word count
        text_lower.count("for"),                    # Loop keywords
        text_lower.count("while"),
        text_lower.count("if"),                     # Conditionals
        text_lower.count("graph"),                  # Data structures
        text_lower.count("tree"),
        text_lower.count("dp"),
        text_lower.count("dynamic programming"),
        text_lower.count("recursion"),              # Algorithms
        text_lower.count("backtracking"),
        text_lower.count("binary search"),
        text_lower.count("shortest path"),
        text_lower.count("mod"),                    # Math operations
        text_lower.count("%"),
        sum(c.isdigit() for c in text),             # Number count
        text_lower.count("+") + text_lower.count("-") + 
        text_lower.count("*") + text_lower.count("/")  # Arithmetic operators
    ]
    
    return features

def get_difficulty_color(label):
    """Get color based on difficulty level"""
    label_lower = label.lower()
    if 'easy' in label_lower:
        return "status-easy"
    elif 'medium' in label_lower:
        return "status-medium"
    elif 'hard' in label_lower:
        return "status-hard"
    else:
        return ""

def display_results(class_label, score_pred):
    """Display prediction results in a styled format"""
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="result-label">DIFFICULTY CLASS</div>', unsafe_allow_html=True)
        color_class = get_difficulty_color(class_label)
        st.markdown(f'<div class="result-value {color_class}">{class_label.upper()}</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="result-label">DIFFICULTY SCORE</div>', unsafe_allow_html=True)
        
        # Score visualization
        score_norm = min(10, max(0, score_pred))
        
        st.markdown(f'<div class="result-value">{score_pred:.1f}/10</div>', unsafe_allow_html=True)
        
        # Progress bar
        st.progress(score_norm / 10)
        
    st.markdown('</div>', unsafe_allow_html=True)

def display_sample_problems():
    """Display sample problems in sidebar"""
    st.sidebar.markdown("### 📋 Sample Problems")
    
    samples = {
        "Easy": "Given two integers, return their sum. Input: Two space-separated integers. Output: Their sum.",
        "Medium": "Implement a function to find the longest substring without repeating characters. Input: A string s. Output: Length of longest substring.",
        "Hard": "Given n non-negative integers representing an elevation map, compute how much water it can trap after raining. Input: Array of n integers. Output: Total trapped water."
    }
    
    for difficulty, text in samples.items():
        if st.sidebar.button(f"Load {difficulty} Example", key=f"sample_{difficulty}"):
            # Split the sample text into sections
            parts = text.split("Input:")
            if len(parts) >= 2:
                desc_part = parts[0].strip()
                input_output = parts[1].split("Output:")
                
                if len(input_output) >= 2:
                    st.session_state.desc = desc_part
                    st.session_state.inp = input_output[0].strip()
                    st.session_state.out = input_output[1].strip()
                    st.rerun()

# Initialize session state
if 'desc' not in st.session_state:
    st.session_state.desc = ""
if 'inp' not in st.session_state:
    st.session_state.inp = ""
if 'out' not in st.session_state:
    st.session_state.out = ""

# Main application
def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div class="header">
        <h1>⚖️ AutoJudge</h1>
        <p>Programming Problem Difficulty Predictor</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 About")
        st.info("""
        This tool predicts the difficulty level of programming problems using machine learning.
        
        **How it works:**
        1. Paste problem details
        2. Click Predict
        3. Get difficulty prediction
        
        The model analyzes text patterns and complexity features.
        """)
        
        st.markdown("---")
        display_sample_problems()
        
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.write("Using:")
        st.write("- Text Classification Model")
        st.write("- Regression Model")
        st.write("- TF-IDF Vectorizer")
        
        st.markdown("---")
        if st.button("🔄 Clear All"):
            st.session_state.desc = ""
            st.session_state.inp = ""
            st.session_state.out = ""
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 📝 Enter Problem Details")
        
        # Input cards
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📄 Problem Description</div>', unsafe_allow_html=True)
        desc = st.text_area(
            "Describe the programming problem",
            value=st.session_state.desc,
            height=150,
            placeholder="Example: Given an array of integers, find two numbers that add up to a specific target...",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        col_in1, col_in2 = st.columns(2)
        
        with col_in1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📥 Input Description</div>', unsafe_allow_html=True)
            inp = st.text_area(
                "Input format and constraints",
                value=st.session_state.inp,
                height=120,
                placeholder="Example: First line contains integer n. Second line contains n space-separated integers...",
                label_visibility="collapsed"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_in2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📤 Output Description</div>', unsafe_allow_html=True)
            out = st.text_area(
                "Output format and requirements",
                value=st.session_state.out,
                height=120,
                placeholder="Example: Print 'YES' if possible, 'NO' otherwise...",
                label_visibility="collapsed"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Predict button
        predict_button = st.button(
            "🚀 Predict Difficulty",
            use_container_width=True,
            type="primary"
        )
        
        # Store in session state
        st.session_state.desc = desc
        st.session_state.inp = inp
        st.session_state.out = out
    
    with col2:
        st.markdown("### 💡 Tips")
        st.info("""
        **For better predictions:**
        
        • Include algorithmic terms
        • Mention time constraints
        • Add example inputs/outputs
        • Be specific about requirements
        
        **Common patterns detected:**
        - Loop constructs
        - Data structures
        - Algorithm keywords
        - Mathematical operations
        """)
    
    # Prediction logic
    if predict_button:
        if not desc or not inp or not out:
            st.warning("⚠️ Please fill in all fields before predicting.")
        else:
            with st.spinner("🔍 Analyzing problem complexity..."):
                # Load models
                models = load_models()
                
                if models:
                    try:
                        # Process input
                        full_text = desc + " " + inp + " " + out
                        clean = clean_text(full_text)
                        
                        # Feature extraction
                        tfidf_feat = models['vectorizer'].transform([clean])
                        # Regression only svd implementation
                        tfidf_svd = models['svd'].transform(tfidf_feat)
                        
                        complexity_feat = np.array([extract_complexity_features(clean)])
                        complexity_feat_reg = models['scaler'].transform(complexity_feat)
                        complexity_feat_clf= csr_matrix(complexity_feat)
                        
                        # Combine features
                        X_input = hstack([tfidf_feat, complexity_feat_clf])
                        X_reg = np.hstack([tfidf_svd, complexity_feat_reg])
                        
                        # Make predictions
                        class_pred = models['clf'].predict(X_input)[0]
                        class_label = models['label_encoder'].inverse_transform([class_pred])[0]
                        score_pred = models['reg'].predict(X_reg)[0]
                        
                        # Clamp score between 0-10 for display
                        score_pred = max(0, min(10, score_pred))
                        
                        # Display results
                        st.markdown("## 📊 Prediction Results")
                        display_results(class_label, score_pred)
                        
                        # Show insights
                        st.markdown("### 🔍 Insights")
                        col_ins1, col_ins2, col_ins3 = st.columns(3)
                        
                        with col_ins1:
                            st.metric("Text Length", f"{len(clean)} chars")
                        
                        with col_ins2:
                            word_count = len(clean.split())
                            st.metric("Word Count", word_count)
                        
                        with col_ins3:
                            st.metric("Complexity Features", len(extract_complexity_features(clean)))
                        
                    except Exception as e:
                        st.error(f"❌ Prediction error: {str(e)}")
                else:
                    st.error("❌ Failed to load models. Please check if model files exist.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        Developed by Sumit Sharma
        <a href="https://github.com/Doofenshmirtz16/acm-open-project-25-26" target="_blank">View on GitHub</a>
    </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()