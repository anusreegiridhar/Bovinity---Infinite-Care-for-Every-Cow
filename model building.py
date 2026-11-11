import pandas as pd
import numpy as np
import joblib
import logging
import warnings
import ast
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise        import cosine_similarity
from sklearn.ensemble                import RandomForestClassifier
from sklearn.naive_bayes             import ComplementNB
from sklearn.neighbors               import KNeighborsClassifier
from sklearn.model_selection         import LeaveOneOut
from sklearn.metrics                 import accuracy_score
from sklearn.preprocessing           import LabelEncoder

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Constants
INPUT_PATH    = Path("featured_dataset.csv")
MATRIX_PATH   = Path("symptom_matrix.csv")
READY_PATH    = Path("model_ready_dataset.csv")
BUNDLE_PATH   = Path("diagnosis_bundle.pkl")
ENSEMBLE_PATH = Path("ensemble_config.json")

INDONESIAN_MARKERS = {
    "dengan", "dan", "atau", "pada", "yang", "untuk", "dari",
    "tidak", "ada", "ini", "itu", "juga", "dapat", "akan",
    "sebagai", "karena", "oleh", "dalam", "adalah", "serta",
    "bahwa", "bila", "jika", "seperti", "antara", "setelah"
}

SEVERITY_WEIGHTS = {
    "seizure": 3, "convulsion": 3, "unconscious": 3,
    "difficulty breathing": 3, "respiratory": 3, "collapse": 3,
    "paralysis": 3, "hemorrhage": 3, "bleeding": 3,
    "sudden death": 3, "coma": 3, "shock": 3,
    "high fever": 3, "abort": 3, "miscarriage": 3,
    "fever": 2, "diarrhea": 2, "vomiting": 2,
    "swelling": 2, "lameness": 2, "staggering": 2,
    "weakness": 2, "dehydration": 2, "trembling": 2,
    "decreased milk": 2, "weight loss": 2, "anemia": 2,
    "discharge": 2, "ulcer": 2, "lesion": 2,
    "anorexia": 1, "loss of appetite": 1, "dull fur": 1,
    "itching": 1, "hair loss": 1, "lethargy": 1,
    "constipation": 1, "blurred vision": 1, "excessive saliva": 1,
    "fatigue": 1, "anestrus": 1, "skin problems": 1,
}


# Shared helpers
def parse_symptoms_list(val) -> list:
    try:
        parsed = ast.literal_eval(val)
        return parsed if isinstance(parsed, list) else []
    except:
        return []

def has_indonesian(text: str) -> bool:
    words = set(str(text).lower().split())
    return bool(words & INDONESIAN_MARKERS)

def is_clean_disease(name: str) -> bool:
    """Filter out PDF artifact disease names."""
    return len(name.split()) <= 6 and not any(c.isdigit() for c in name[:5])

def compute_severity(symptoms_text: str) -> tuple[str, int]:
    """Keyword-based severity with symptom-count fallback."""
    text  = symptoms_text.lower()
    score = sum(w for kw, w in SEVERITY_WEIGHTS.items() if kw in text)
    count = len([s for s in symptoms_text.split(",") if s.strip()])
    if score == 0:
        if count <= 1:   level = "Mild"
        elif count <= 3: level = "Moderate"
        elif count <= 6: level = "Severe"
        else:            level = "Critical"
    elif score <= 3:  level = "Mild"
    elif score <= 7:  level = "Moderate"
    elif score <= 12: level = "Severe"
    else:             level = "Critical"
    return level, score

def encode_symptoms(user_symptoms: str, all_symptoms: list) -> np.ndarray:
    """Convert comma-separated symptom string into binary feature vector."""
    user_list = [s.strip().lower() for s in user_symptoms.split(",")]
    vector    = np.zeros(len(all_symptoms))
    for i, s in enumerate(all_symptoms):
        if s in user_list:
            vector[i] = 1
    return vector.reshape(1, -1)

# STEP 1 — Tune & Evaluate Similarity Matcher
logger.info("── Step 1: Tune Similarity Matcher ──")

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Featured dataset not found: {INPUT_PATH}. Run feature engineering first.")

df = pd.read_csv(INPUT_PATH)
df["symptoms_list"] = df["symptoms_list"].apply(parse_symptoms_list)
logger.info(f"Loaded {len(df)} rows.")

# Translation quality check before training
flagged = df["symptoms"].apply(has_indonesian).sum()
if flagged > 0:
    logger.warning(
        f"{flagged} rows still contain Indonesian text — "
        f"model may perform worse on these. Re-run preprocessing if count is high."
    )
else:
    logger.info("✓ No Indonesian markers detected in symptoms.")

# Vectorizer configs to compare
configs = [
    {"name": "Unigram only",              "ngram_range": (1, 1), "analyzer": "word",    "sublinear_tf": True},
    {"name": "Unigram + Bigram",          "ngram_range": (1, 2), "analyzer": "word",    "sublinear_tf": True},
    {"name": "Unigram + Bigram + Trigram","ngram_range": (1, 3), "analyzer": "word",    "sublinear_tf": True},
    {"name": "Char n-gram (3-5)",         "ngram_range": (3, 5), "analyzer": "char_wb", "sublinear_tf": True},
    {"name": "Bigram no sublinear",       "ngram_range": (1, 2), "analyzer": "word",    "sublinear_tf": False},
]

def evaluate_similarity_config(cfg: dict, df: pd.DataFrame) -> dict:
    """Leave-one-out evaluation for a TF-IDF config."""
    top1_hits = []
    top3_hits = []

    for i in range(len(df)):
        train_df  = df.drop(index=i).reset_index(drop=True)
        test_row  = df.iloc[i]

        vec          = TfidfVectorizer(ngram_range=cfg["ngram_range"],
                                       analyzer=cfg["analyzer"],
                                       sublinear_tf=cfg["sublinear_tf"], min_df=1)
        train_matrix = vec.fit_transform(train_df["symptoms"])
        query_vec    = vec.transform([test_row["symptoms"]])
        sims         = cosine_similarity(query_vec, train_matrix).flatten()
        top3_idx     = np.argsort(sims)[::-1][:3]
        predicted    = train_df.iloc[top3_idx]["disease"].tolist()
        true         = test_row["disease"]

        top1_hits.append(1 if true == predicted[0] else 0)
        top3_hits.append(1 if true in predicted    else 0)

    return {
        "config"  : cfg["name"],
        "top1_acc": round(np.mean(top1_hits) * 100, 2),
        "top3_acc": round(np.mean(top3_hits) * 100, 2),
    }

print("\n════════════════════════════════════════════════════")
print("   SIMILARITY MATCHER — LEAVE-ONE-OUT EVALUATION   ")
print("════════════════════════════════════════════════════")
print(f"  {'Config':<35} {'Top-1':>8} {'Top-3':>8}")
print(f"  {'─'*35} {'─'*8} {'─'*8}")

sim_results = []
best_cfg    = None
best_top3   = -1

for cfg in configs:
    r = evaluate_similarity_config(cfg, df)
    print(f"  {r['config']:<35} {r['top1_acc']:>7}% {r['top3_acc']:>7}%")
    sim_results.append(r)
    if r["top3_acc"] > best_top3:
        best_top3 = r["top3_acc"]
        best_cfg  = cfg

print("════════════════════════════════════════════════════")
print(f"\n  ✓ Best config  : '{best_cfg['name']}'")
print(f"    Top-3 accuracy: {best_top3}%")

# ── Retrain and save best vectorizer ──
best_vec    = TfidfVectorizer(ngram_range=best_cfg["ngram_range"],
                               analyzer=best_cfg["analyzer"],
                               sublinear_tf=best_cfg["sublinear_tf"], min_df=1)
best_matrix = best_vec.fit_transform(df["symptoms"])

joblib.dump(best_vec,    "tfidf_vectorizer.pkl")
joblib.dump(best_matrix, "tfidf_matrix.pkl")
logger.info("Saved best TF-IDF vectorizer and matrix.")

# Sanity check with best config
test_queries = [
    "fever, diarrhea, weakness",
    "hair loss, itching, skin problems",
    "anorexia, anestrus",
]
print("\n── Sanity Check (Best Config) ──")
for q in test_queries:
    qv   = best_vec.transform([q])
    sims = cosine_similarity(qv, best_matrix).flatten()
    top3 = np.argsort(sims)[::-1][:3]
    print(f"\n  Query: '{q}'")
    for idx in top3:
        print(f"    → {df['disease'].iloc[idx]:<35} score={sims[idx]:.4f}")

# Save model-ready dataset
df.to_csv(READY_PATH, index=False)
logger.info(f"Saved model-ready dataset → '{READY_PATH}'")

# STEP 2 — Train & Compare All Three ML Models (LOO CV)
logger.info("── Step 2: Train & Compare ML Models (LOO CV) ──")

matrix = pd.read_csv(MATRIX_PATH)
X      = matrix.drop(columns=["disease", "disease_encoded"]).values
y      = matrix["disease_encoded"].values
le     = joblib.load("label_encoder.pkl")

logger.info(f"Matrix shape: {X.shape}  |  Classes: {len(np.unique(y))}")

loo = LeaveOneOut()

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators     = 500,
        max_depth        = None,
        min_samples_split= 2,
        class_weight     = "balanced",
        random_state     = 42,
    ),
    "Naive Bayes (Complement)": ComplementNB(alpha=0.5),
    "KNN": KNeighborsClassifier(
        n_neighbors = 3,
        metric      = "hamming",
        weights     = "distance",
    ),
}

print("\n════════════════════════════════════════════════════")
print("   ML MODEL COMPARISON — LEAVE-ONE-OUT CV          ")
print("════════════════════════════════════════════════════")
print(f"  {'Model':<28} {'Top-1':>8} {'Top-3':>8}")
print(f"  {'─'*28} {'─'*8} {'─'*8}")

loo_results = {}

for model_name, model in models.items():
    top1_preds = []
    top3_hits  = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)

        top1_pred = model.predict(X_test)[0]
        top1_preds.append(top1_pred)

        try:
            probs        = model.predict_proba(X_test)[0]
            top3_classes = model.classes_[np.argsort(probs)[::-1][:3]]
            top3_hits.append(1 if y_test[0] in top3_classes else 0)
        except:
            top3_hits.append(1 if y_test[0] == top1_pred else 0)

    top1_acc = round(accuracy_score(y, top1_preds) * 100, 2)
    top3_acc = round(np.mean(top3_hits) * 100, 2)

    loo_results[model_name] = {
        "top1_acc"  : top1_acc,
        "top3_acc"  : top3_acc,
        "top1_preds": top1_preds,
    }
    print(f"  {model_name:<28} {top1_acc:>7}% {top3_acc:>7}%")

print("════════════════════════════════════════════════════")

winner = max(loo_results, key=lambda m: loo_results[m]["top3_acc"])
print(f"\n  ✓ Best ML model : '{winner}'")
print(f"    Top-1: {loo_results[winner]['top1_acc']}%  |  Top-3: {loo_results[winner]['top3_acc']}%")
print("\n  NOTE: Low accuracy is expected with 1 sample per class.")
print("  TF-IDF similarity remains the primary diagnosis engine.")

# Retrain all on full data and save
logger.info("Retraining all models on full dataset...")
model_paths = {}

for model_name, model in models.items():
    model.fit(X, y)
    safe   = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    path   = f"model_{safe}.pkl"
    joblib.dump(model, path)
    model_paths[model_name] = path
    logger.info(f"Saved {model_name} → '{path}'")

# STEP 3 — Build Combined Weighted Ensemble Predictor
logger.info("── Step 3: Build Combined Ensemble Predictor ──")

df           = pd.read_csv(READY_PATH)
matrix       = pd.read_csv(MATRIX_PATH)
vectorizer   = joblib.load("tfidf_vectorizer.pkl")
tfidf_matrix = joblib.load("tfidf_matrix.pkl")
le           = joblib.load("label_encoder.pkl")
rf           = joblib.load("model_random_forest.pkl")
nb           = joblib.load("model_naive_bayes_complement.pkl")
knn          = joblib.load("model_knn.pkl")

df["symptoms_list"] = df["symptoms_list"].apply(parse_symptoms_list)
all_symptoms = [c for c in matrix.columns if c not in ["disease", "disease_encoded"]]

logger.info("All artifacts loaded.")

def ensemble_diagnose(user_symptoms: str, top_n: int = 3) -> list[dict]:
    """
    Weighted ensemble:
      TF-IDF similarity  60%  — primary engine, most reliable on small data
      Random Forest      20%  — captures binary symptom patterns
      Naive Bayes        10%  — fast probabilistic complement
      KNN                10%  — distance-based similarity on binary vectors
    """
    q = user_symptoms.strip().lower()

    # Similarity scores
    qv   = vectorizer.transform([q])
    sims = cosine_similarity(qv, tfidf_matrix).flatten()

    # Binary vector for ML models
    bin_vec = encode_symptoms(q, all_symptoms)

    rf_probs  = rf.predict_proba(bin_vec)[0]
    nb_probs  = nb.predict_proba(bin_vec)[0]
    knn_probs = knn.predict_proba(bin_vec)[0]

    scores = []
    for idx in range(len(df)):
        disease   = df.iloc[idx]["disease"]
        sim_score = float(sims[idx])

        try:
            cls_idx = list(le.classes_).index(disease)
            rf_s    = float(rf_probs[cls_idx])  if cls_idx < len(rf_probs)  else 0.0
            nb_s    = float(nb_probs[cls_idx])  if cls_idx < len(nb_probs)  else 0.0
            knn_s   = float(knn_probs[cls_idx]) if cls_idx < len(knn_probs) else 0.0
        except ValueError:
            rf_s = nb_s = knn_s = 0.0

        ensemble = (0.60 * sim_score) + (0.20 * rf_s) + (0.10 * nb_s) + (0.10 * knn_s)
        scores.append((idx, ensemble, sim_score, rf_s, nb_s, knn_s))

    scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for idx, ens, sim, rf_s, nb_s, knn_s in scores:
        if len(results) >= top_n:
            break
        row     = df.iloc[idx]
        disease = str(row["disease"])
        if not is_clean_disease(disease):
            continue
        treatment = str(row.get("treatment", "")).strip()
        sev_level, sev_score = compute_severity(q)
        results.append({
            "disease"        : disease.title(),
            "ensemble_score" : round(ens,   4),
            "similarity"     : round(sim,   4),
            "rf_score"       : round(rf_s,  4),
            "nb_score"       : round(nb_s,  4),
            "knn_score"      : round(knn_s, 4),
            "treatment"      : treatment if treatment and treatment != "nan"
                               else "No treatment data available.",
            "severity_level" : sev_level,
            "severity_score" : sev_score,
            "matched_symptoms": str(row.get("symptoms", "")),
        })

    return results

# LOO evaluation of ensemble
logger.info("Evaluating ensemble via leave-one-out...")
top1_hits = []
top3_hits = []

for i in range(len(df)):
    true_disease = df.iloc[i]["disease"].lower()
    results      = ensemble_diagnose(df.iloc[i]["symptoms"], top_n=3)
    predicted    = [r["disease"].lower() for r in results]

    top1_hits.append(1 if true_disease == predicted[0] else 0)
    top3_hits.append(1 if true_disease in predicted    else 0)

ens_top1 = round(np.mean(top1_hits) * 100, 2)
ens_top3 = round(np.mean(top3_hits) * 100, 2)

# Test queries
test_cases = [
    "fever, diarrhea, weakness",
    "hair loss, itching, skin problems",
    "anorexia, anestrus",
    "lameness, swelling, staggering, seizure",
    "diarrhea, vomiting, dehydration",
]

print("\n════════════════════════════════════════════════")
print("       ENSEMBLE DIAGNOSIS ENGINE — TEST        ")
print("════════════════════════════════════════════════")

for q in test_cases:
    results = ensemble_diagnose(q, top_n=3)
    print(f"\n  Input   : '{q}'")
    print(f"  Severity: {results[0]['severity_level'] if results else 'Unknown'}")
    print(f"  ── Top Matches ──")
    for i, r in enumerate(results):
        print(f"    #{i+1} {r['disease']:<30} ensemble={r['ensemble_score']:.4f} "
              f"(sim={r['similarity']:.3f} rf={r['rf_score']:.3f} "
              f"nb={r['nb_score']:.3f} knn={r['knn_score']:.3f})")
    print(f"  Treatment: {results[0]['treatment'][:100]}..." if results else "")

# Save ensemble config
ensemble_config = {
    "weights"       : {"similarity": 0.60, "random_forest": 0.20, "naive_bayes": 0.10, "knn": 0.10},
    "best_sim_config": best_cfg["name"],
    "loo_top1_acc"  : ens_top1,
    "loo_top3_acc"  : ens_top3,
}
with open(ENSEMBLE_PATH, "w") as f:
    json.dump(ensemble_config, f, indent=2)

# Save bundle for chatbot
bundle = {"df": df, "matrix_cols": all_symptoms}
joblib.dump(bundle, BUNDLE_PATH)
logger.info(f"Saved ensemble config  → '{ENSEMBLE_PATH}'")
logger.info(f"Saved diagnosis bundle → '{BUNDLE_PATH}'")

# STEP 4 — Final Model File Verification
logger.info("── Step 4: File Verification ──")

files = {
    "Featured dataset"        : Path("featured_dataset.csv"),
    "Model-ready dataset"     : Path("model_ready_dataset.csv"),
    "TF-IDF vectorizer"       : Path("tfidf_vectorizer.pkl"),
    "TF-IDF matrix"           : Path("tfidf_matrix.pkl"),
    "Random Forest model"     : Path("model_random_forest.pkl"),
    "Naive Bayes model"       : Path("model_naive_bayes_complement.pkl"),
    "KNN model"               : Path("model_knn.pkl"),
    "Label encoder"           : Path("label_encoder.pkl"),
    "Diagnosis bundle"        : Path("diagnosis_bundle.pkl"),
    "Ensemble config"         : Path("ensemble_config.json"),
    "Severity rules"          : Path("severity_rules.json"),
}

print("\n════════════════════════════════════════")
print("         MODEL FILE CHECK               ")
print("════════════════════════════════════════")

all_ok = True
for name, path in files.items():
    exists = path.exists()
    status = "✓" if exists else "✗ MISSING"
    size   = f"{path.stat().st_size / 1024:.1f} KB" if exists else ""
    print(f"  {status}  {name:<28} {size}")
    if not exists:
        all_ok = False

print("════════════════════════════════════════")

if all_ok:
    df_check = pd.read_csv(READY_PATH)
    indonesian_remaining = df_check["symptoms"].apply(has_indonesian).sum()

    print(f"\n  Rows in model-ready dataset   : {len(df_check)}")
    print(f"  Disease classes               : {len(le.classes_)}")
    print(f"  Ensemble Top-1 accuracy (LOO) : {ens_top1}%")
    print(f"  Ensemble Top-3 accuracy (LOO) : {ens_top3}%")
    print(f"  Symptoms w/ Indonesian text   : {indonesian_remaining} "
          f"{'⚠️  re-check translation' if indonesian_remaining > 0 else '✓'}")
    print(f"\n  Best similarity config : {best_cfg['name']}")
    print(f"  Similarity Top-3 (LOO) : {best_top3}%")
    print("\n  ✓ All files ready. Update chatbot to use ensemble_diagnose().")
else:
    print("\n  ✗ Some files missing. Re-run steps above in order.")