# Bovinity — Infinite Care for Every Cow

**Bovinity** is an intelligent livestock healthcare platform powered by Natural Language Processing (NLP) and Machine Learning (ML). It converts unstructured multi-lingual veterinary records into structured symptom mappings to assist farmers and livestock handlers in identifying cattle diseases, assessing severity, and obtaining preliminary treatment recommendations.

## Key Features

* **Multilingual Neural Machine Translation (NMT):** Translates raw Indonesian veterinary PDF text into standardized English using Hugging Face's `MarianMT` (`Helsinki-NLP/opus-mt-id-en`).
* **Text Preprocessing & Cleanup:** Automated PDF scraping via `pdfplumber`, prefix artifact removal, noise token filtering, and token normalization.
* **Feature Extraction:** Transforms unstructured clinical descriptions into sparse **TF-IDF n-gram vectors** (unigrams and bigrams) and binary symptom matrices.
* **Hybrid Diagnostic Engine:** Combines a 4-component weighted model:
  * **60%** Cosine Similarity over TF-IDF vectors (primary retrieval engine)
  * **20%** Random Forest Classifier
  * **10%** Complement Naive Bayes Classifier
  * **10%** K-Nearest Neighbors (KNN) Classifier
* **Dynamic Severity Classifier:** Scans user inputs against clinical severity lexicons to assign risk levels (**Mild**, **Moderate**, **Severe**, **Critical**).
* **Streamlit Dashboard:** Interactive web app with real-time symptom token validation, input chip previews, unknown token alerts, and diagnostic history tracking.

---

## Architecture & Pipeline

```text
┌──────────────────────────────────────┐
│       Raw Indonesian PDF Data        │
└──────────────────┬───────────────────┘
                   │
                   ▼ (pdfplumber + Regex Parsing)
┌──────────────────────────────────────┐
│     Structured Field Extraction      │
└──────────────────┬───────────────────┘
                   │
                   ▼ (MarianMT: Helsinki-NLP/opus-mt-id-en)
┌──────────────────────────────────────┐
│     Batch Offline NMT Translation    │
└──────────────────┬───────────────────┘
                   │
                   ▼ (Tokenization & Filtering)
┌──────────────────────────────────────┐
│    Feature Vectorization & Matrix    │
│    - TF-IDF (1,2 N-grams)            │
│    - Binary Symptom Matrix           │
└──────────────────┬───────────────────┘
                   │
                   ▼ (Ensemble Predictor & Severity Scorer)
┌──────────────────────────────────────┐
│          Diagnosis Engine            │
│  - Cosine Similarity (60%)           │
│  - Random Forest (20%)               │
│  - Complement Naive Bayes (10%)      │
│  - K-Nearest Neighbors (10%)         │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│       Streamlit Interactive UI       │
└──────────────────────────────────────┘
```

---

## Repository Structure

```text
.
├── dataset 2.pdf                                      # Raw Indonesian veterinary dataset
├── data science mini project dataset 2 preprocessing.py # PDF extraction & MarianMT translation
├── eda.py.py                                          # Exploratory Data Analysis & quality checks
├── feature engineering.py                             # Matrix creation, TF-IDF & severity rules
├── model training.py                                  # Model tuning, LOO CV & ensemble export
├── chatbot interface.py                               # Streamlit web app
├── model_ready_dataset.csv                            # Cleaned & processed dataset
├── symptom_matrix.csv                                 # Binary symptom feature matrix
├── tfidf_vectorizer.pkl                               # Saved TF-IDF vectorizer
├── tfidf_matrix.pkl                                   # Pre-computed TF-IDF matrix
├── model_random_forest.pkl                            # Trained Random Forest model
├── label_encoder.pkl                                  # Label encoder for diseases
├── diagnosis_bundle.pkl                               # Exported diagnosis bundle
├── ensemble_config.json                               # Weight configurations & metadata
├── severity_rules.json                                # Clinical keyword severity rules
└── README.md                                          # Project Documentation
```

---

## 🛠️ Tech Stack

* **Language:** Python 3.13
* **NLP & NMT:** `transformers` (MarianMT), `scikit-learn` (`TfidfVectorizer`), `nltk`, `re`
* **Machine Learning:** `scikit-learn` (`RandomForestClassifier`, `ComplementNB`, `KNeighborsClassifier`, `cosine_similarity`)
* **PDF Extraction:** `pdfplumber`
* **Web Interface:** `streamlit`
* **Data & Analytics:** `pandas`, `numpy`, `joblib`, `matplotlib`, `seaborn`

---

## Getting Started

```bash
# 1. Extract & translate PDF dataset
python "data science mini project dataset 2 preprocessing.py"

# 2. Run Exploratory Data Analysis
python "eda.py.py"

# 3. Feature engineering & TF-IDF vectorization
python "feature engineering.py"

# 4. Train ensemble models & evaluate (Leave-One-Out CV)
python "model training.py"

# 5. Launch the Web Application
streamlit run "chatbot interface.py"
```

---

## Model Evaluation & Logic

* **Retrieval Logic:** TF-IDF Cosine Similarity forms the 60% core weight of the diagnosis engine to maintain high precision even for sparse, single-sample disease classes.
* **Cross-Validation:** Evaluated using **Leave-One-Out (LOO) Cross-Validation**.
* **Severity Scoring Weights:**
  * **Critical (+3):** Seizure, paralysis, hemorrhage, collapse, respiratory distress, coma.
  * **Severe/Moderate (+2):** Fever, diarrhea, lameness, staggering, swelling, dehydration.
  * **Mild (+1):** Anorexia, dull fur, itching, lethargy, skin problems.

---

## Disclaimer

Bovinity provides **preliminary educational guidance only** based on statistical symptom matching. It is not a replacement for professional veterinary clinical diagnosis. Always consult a licensed veterinarian before administering medicine or performing medical treatment on livestock.

---
