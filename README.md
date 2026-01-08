# AutoJudge: Predicting Programming Problem Difficulty

**AutoJudge** is a machine learning–based system that automatically predicts the **difficulty level** of programming problems using **only their textual descriptions**.  
The project addresses the subjectivity and inconsistency in manual difficulty assignment on online coding platforms by building a data-driven, reproducible solution.

The system predicts:
- **Difficulty Class** → Easy / Medium / Hard *(Classification)*
- **Difficulty Score** → Numerical value on a continuous scale *(Regression)*

A simple and interactive **web interface** allows users to paste a new problem description and instantly receive predictions.

---

## Project Overview

Online competitive programming platforms (e.g., Codeforces, CodeChef) typically assign problem difficulty based on human judgment and user feedback.  
However, such methods are subjective and require large-scale participation.

**AutoJudge** aims to:
- Predict problem difficulty **automatically**
- Use **only textual information**
- Avoid reliance on user statistics or metadata
- Provide both **categorical** and **continuous** difficulty estimates

The project combines **classical NLP techniques**, **feature engineering**, and **machine learning models**, without using deep learning.

---

## Dataset Used

- **Format:** JSON Lines (`.jsonl`)
- **Source:** Provided dataset (reference link supplied in problem statement)
- **Each data sample contains:**
  - `title`
  - `description`
  - `input_description`
  - `output_description`
  - `problem_class` (Easy / Medium / Hard)
  - `problem_score` (Numerical difficulty)

> Note: No manual labeling or external data was used.

---

## Approach & Methodology

### 1. Data Preprocessing
- Combined all text fields into a single unified text input
- Cleaned text while **preserving programming symbols** (`+`, `*`, `<=`, etc.)
- Handled missing values safely using empty strings

---

### 2. Feature Engineering

#### 🔹 Textual Features (Core)
- **TF-IDF Vectorization**
  - Unigrams + Bigrams
  - Stopword removal
  - Sparse representation

#### 🔹 Handcrafted Complexity Features
Extracted structural indicators such as:
- Text length, word count
- Loop keywords (`for`, `while`)
- Conditional keywords (`if`)
- Algorithmic hints (`graph`, `tree`, `dp`)
- Numeric and arithmetic operator density

#### 🔹 Dimensionality Reduction (Regression Only)
- **Truncated SVD (Latent Semantic Analysis)**
- Applied **only to regression pipeline**
- Reduced sparsity and captured latent difficulty patterns

---

## Models Used

### 🔸 Model 1: Difficulty Classification
- **Random Forest Classifier**
- Predicts: Easy / Medium / Hard
- Chosen for robustness to noisy text features
- No dimensionality reduction applied

### 🔸 Model 2: Difficulty Score Regression
- **Gradient Boosting Regressor**
- Predicts continuous difficulty score
- Uses:
  - TF-IDF → SVD
  - Scaled complexity features
- Achieved best performance after extensive experimentation

---

## Evaluation Metrics & Results

### 🔹 Classification Performance
- **Accuracy:** ~ **0.54**
- Confusion mostly between *Medium* and *Hard* (expected)
- Classification treated as **coarse-grained guidance**

### 🔹 Regression Performance (Final Model)
| Metric | Value |
|-----|------|
| **MAE** | **≈ 1.68** |
| **RMSE** | **≈ 2.02** |

> Regression significantly outperformed classification, indicating that difficulty is better modeled as a **continuous variable**.

---

## Web Interface Explanation

The project includes a **Streamlit-based web app** that allows users to:

### Input
- Problem Description
- Input Description
- Output Description

### Output
- Predicted Difficulty Class (Easy / Medium / Hard)
- Predicted Difficulty Score (0–10 scale)
- Visual indicators (progress bar, color-coded labels)

### Features
- Clean UI with instructions
- Sample problems for testing
- Cached models for fast inference
- Consistent preprocessing between training and inference

---

## ▶️ Steps to Run the Project Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Doofenshmirtz16/acm-open-project-25-26.git
cd acm-open-project-25-26
```

### 2. Create & Activate Virtual Environment
```bash
python -m venv autojudge_env
source autojudge_env/bin/activate   # Linux/Mac
autojudge_env\Scripts\activate      # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Web App
```bash
streamlit run app.py
```
- The app will open in your browser at localhost.

---

## Demo Video

- 2–3 minute project demo: https://drive.google.com/file/d/1Iw-RoB5Skc-KKm6WE_mu_CNgdfCMPruw/view?usp=sharing

---

## Project Structure
```bash
AutoJudge/
│
├── data/
│   └── problems_data.jsonl
│
├── notebooks/
│   └── final_notebook.ipynb
│
├── models/
│   ├── classification_model.pkl
│   ├── final_regression_model.pkl
│   ├── vectorizer.pkl
│   ├── svd.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Limitations
- Difficulty labels are inherently subjective
- True difficulty depends on constraints and hidden test cases
- Uses only textual data (no runtime or submission statistics)

---

## Future Improvements
- Ordinal classification models
- Model explainability (SHAP)
- Cross-platform deployment (Streamlit Cloud / Hugging Face)
- Support for additional problem metadata (if allowed)

---

## Author Details
- Name: Sumit Sharma
- Project: ACM Open Project 2025–26
- GitHub: https://github.com/Doofenshmirtz16
