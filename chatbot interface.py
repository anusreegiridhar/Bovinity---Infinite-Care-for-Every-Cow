import streamlit as st
import pandas as pd
import numpy as np
import joblib
import ast
import re
from sklearn.metrics.pairwise import cosine_similarity

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bovinity — Infinite Care for Every Cow",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Lilita+One&family=Nunito:wght@400;600;700;800&family=Lora:ital@0;1&display=swap');

  :root {
    --brown:       #ce9071;
    --brown-dark:  #8b4513;
    --brown-deep:  #5c2a0a;
    --brown-light: #f2ddd0;
    --cream:       #fdf8f5;
    --white:       #ffffff;
    --text:        #3d1a0e;
    --text-light:  #7a4a30;
    --red:         #c0392b;
    --orange:      #d4600a;
    --yellow:      #a07800;
    --green:       #1a7a4a;
  }

  html, body, [class*="css"] { font-family: 'Nunito', sans-serif; color: var(--text); }

  .stApp {
    background-color: var(--white);
    background-image:
      radial-gradient(ellipse 220px 170px at  2%   4%,  #ce9071 99%, transparent 100%),
      radial-gradient(ellipse 160px 200px at 14%   0%,  #ce9071 99%, transparent 100%),
      radial-gradient(ellipse 140px 110px at  0%  18%,  #ce9071 99%, transparent 100%),
      radial-gradient(ellipse 240px 160px at 98%   3%,  #ce9071 99%, transparent 100%),
      radial-gradient(ellipse 170px 220px at 86%   0%,  #ce9071 99%, transparent 100%),
      radial-gradient(ellipse 130px 160px at 100% 20%,  #ce9071 99%, transparent 100%),
      radial-gradient(ellipse 180px 140px at  0%  48%,  #ce9071 99%, transparent 100%),
      radial-gradient(ellipse 120px 180px at  4%  62%,  #ce9071 99%, transparent 100%),
      radial-gradient(ellipse 200px 150px at 100% 45%,  #ce9071 99%, transparent 100%),
      radial-gradient(ellipse 150px 190px at  96% 60%,  #ce9071 99%, transparent 100%),
      radial-gradient(ellipse 210px 160px at  3%  94%,  #ce9071 99%, transparent 100%),
      radial-gradient(ellipse 160px 130px at 16%  99%,  #ce9071 99%, transparent 100%),
      radial-gradient(ellipse 200px 170px at 97%  92%,  #ce9071 99%, transparent 100%),
      radial-gradient(ellipse 150px 140px at 80%  99%,  #ce9071 99%, transparent 100%),
      radial-gradient(ellipse 130px 100px at 50% 100%,  #ce9071 99%, transparent 100%),
      radial-gradient(ellipse 120px  90px at 50%   0%,  #ce9071 99%, transparent 100%);
    background-attachment: fixed;
  }

  [data-testid="stSidebar"] {
    background: linear-gradient(175deg, #3d1a0e 0%, #6b3020 55%, #8b4513 100%) !important;
    border-right: 4px solid var(--brown) !important;
  }
  [data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
  [data-testid="stSidebar"] * { color: #fde8d8 !important; }
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3,
  [data-testid="stSidebar"] h4 { color: #fff !important; }
  [data-testid="stSidebar"] hr { border-color: rgba(206,144,113,0.4) !important; margin: 12px 0 !important; }

  .main .block-container {
    background: rgba(255,255,255,0.88);
    border-radius: 28px;
    padding: 2rem 2.8rem 3rem !important;
    backdrop-filter: blur(8px);
    box-shadow: 0 10px 48px rgba(139,69,19,0.12), 0 2px 8px rgba(139,69,19,0.06);
    margin-top: 1rem;
    max-width: 900px;
  }

  .hero-outer {
    display: flex; align-items: center; justify-content: center;
    gap: 32px; padding: 28px 0 10px;
  }
  .hero-left { display: flex; flex-direction: column; align-items: flex-start; }
  .hero-title {
    font-family: 'Lilita One', cursive;
    font-size: 40px; letter-spacing: 4px; color: var(--brown);
    -webkit-text-stroke: 2.5px var(--brown-dark);
    text-shadow: 3px 3px 0 var(--brown-dark), 1px 1px 0 rgba(255,255,255,0.6);
    margin: 0; line-height: 1.1;
  }
  .hero-tagline {
    font-family: 'Lora', serif; font-style: italic; font-size: 20px;
    color: var(--brown-dark); margin: 8px 0 0 3px; line-height: 1.4;
  }
  .hero-logo {
    font-size: 88px; line-height: 1;
    filter: drop-shadow(3px 4px 6px rgba(139,69,19,0.25));
    animation: moo 4s ease-in-out infinite;
  }
  @keyframes moo {
    0%, 100% { transform: rotate(-3deg) scale(1); }
    50%       { transform: rotate(3deg)  scale(1.06); }
  }

  .sec-label {
    font-family: 'Nunito', sans-serif; font-weight: 800; font-size: 12px;
    letter-spacing: 1.8px; text-transform: uppercase; color: var(--brown-dark);
    margin: 28px 0 10px; display: flex; align-items: center; gap: 6px;
  }
  .sec-label::after {
    content: ''; flex: 1; height: 2px;
    background: linear-gradient(90deg, var(--brown-light), transparent);
    border-radius: 2px;
  }

  .stTextInput > div > div > input {
    border: 2.5px solid var(--brown) !important;
    border-radius: 16px !important; font-family: 'Nunito', sans-serif !important;
    font-size: 16px !important; padding: 14px 18px !important;
    background: var(--white) !important; color: var(--text) !important;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  .stTextInput > div > div > input:focus {
    border-color: var(--brown-dark) !important;
    box-shadow: 0 0 0 4px rgba(206,144,113,0.20) !important;
  }
  .stTextInput > div > div > input::placeholder { color: #b08060 !important; }

  .stButton > button {
    background: linear-gradient(135deg, var(--brown) 0%, var(--brown-dark) 100%) !important;
    color: #fff !important; border: none !important; border-radius: 16px !important;
    font-family: 'Nunito', sans-serif !important; font-weight: 800 !important;
    font-size: 16px !important; padding: 13px 0 !important;
    box-shadow: 0 5px 16px rgba(139,69,19,0.32);
    transition: transform 0.15s, box-shadow 0.15s;
  }
  .stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 24px rgba(139,69,19,0.42) !important; }
  .stButton > button:active { transform: translateY(0) !important; }

  .chips-wrap { display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0 6px; }
  .chip {
    background: var(--brown-light); border: 2px solid var(--brown);
    border-radius: 24px; padding: 5px 16px; font-size: 13px;
    color: var(--brown-deep); font-family: 'Nunito', sans-serif; font-weight: 700;
  }
  .chip-unknown {
    background: #fff0f0; border: 2px solid #e8a0a0;
    border-radius: 24px; padding: 5px 16px; font-size: 13px;
    color: #8b2020; font-family: 'Nunito', sans-serif; font-weight: 700;
  }

  .unknown-warn {
    background: #fff5f5; border: 2px solid #e8a0a0;
    border-left: 6px solid #c0392b; border-radius: 14px;
    padding: 16px 20px; font-family: 'Nunito', sans-serif;
    font-size: 15px; font-weight: 600; color: #6b1010;
    margin: 10px 0 16px; line-height: 1.7;
  }
  .unknown-warn b { color: #c0392b; }

  .howto-box {
    background: linear-gradient(135deg, #fdf8f5 60%, #f7ede3);
    border: 2px solid var(--brown-light); border-left: 5px solid var(--brown);
    border-radius: 18px; padding: 20px 24px; font-family: 'Nunito', sans-serif;
    font-size: 15px; color: var(--text); margin-bottom: 24px; line-height: 1.8;
  }
  .howto-step { display: flex; align-items: flex-start; gap: 14px; margin-bottom: 10px; }
  .step-num {
    background: var(--brown); color: white; font-weight: 800; font-size: 14px;
    width: 28px; height: 28px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; margin-top: 2px;
  }

  .bubble-user {
    background: linear-gradient(135deg, var(--brown), var(--brown-dark));
    color: white; border-radius: 22px 22px 6px 22px;
    padding: 14px 20px; margin: 14px 0 8px auto;
    max-width: 75%; width: fit-content;
    font-family: 'Nunito', sans-serif; font-size: 15px; font-weight: 600;
    box-shadow: 0 4px 14px rgba(139,69,19,0.22); line-height: 1.5;
  }
  .bubble-bot {
    background: var(--white); border: 2px solid var(--brown-light);
    border-radius: 22px 22px 22px 6px;
    padding: 14px 20px; margin: 8px auto 14px 0;
    max-width: 75%; width: fit-content;
    font-family: 'Nunito', sans-serif; font-size: 15px; color: var(--text);
    box-shadow: 0 4px 14px rgba(139,69,19,0.08); line-height: 1.6;
  }

  .sev-badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 5px 14px; border-radius: 24px; font-weight: 800; font-size: 13px;
    font-family: 'Nunito', sans-serif;
  }
  .sev-Critical { background: #fde8e8; color: var(--red);    border: 1.5px solid #f5b8b8; }
  .sev-Severe   { background: #ffebd4; color: var(--orange); border: 1.5px solid #f5c898; }
  .sev-Moderate { background: #fff8d4; color: var(--yellow); border: 1.5px solid #f5e098; }
  .sev-Mild     { background: #e0f8ec; color: var(--green);  border: 1.5px solid #98e0c0; }
  .sev-Unknown  { background: #f0f0f0; color: #888;          border: 1.5px solid #ddd; }

  .result-card {
    background: var(--white); border-radius: 18px; padding: 20px 24px; margin: 12px 0;
    box-shadow: 0 3px 14px rgba(139,69,19,0.09); border-left: 6px solid var(--brown);
    font-family: 'Nunito', sans-serif; transition: box-shadow 0.2s;
  }
  .result-card:hover { box-shadow: 0 6px 24px rgba(139,69,19,0.16); }
  .result-card.critical { border-left-color: var(--red);    background: #fff8f8; }
  .result-card.severe   { border-left-color: var(--orange); background: #fffaf4; }
  .result-card.moderate { border-left-color: #d4a000;       background: #fffef0; }
  .result-card.mild     { border-left-color: var(--green);  background: #f4fdf8; }

  .card-title { font-size: 20px; font-weight: 800; color: var(--text); font-family: 'Nunito', sans-serif; }
  .card-pct   { font-size: 26px; font-weight: 900; color: var(--brown-dark); font-family: 'Lilita One', cursive; }
  .card-row   { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }
  .bar-wrap   { margin: 8px 0 14px; }
  .bar-label  { font-size: 11px; color: var(--text-light); font-weight: 700; letter-spacing: 0.5px; margin-bottom: 4px; }
  .bar-bg     { background: var(--brown-light); border-radius: 10px; height: 10px; }
  .bar-fg     { height: 10px; border-radius: 10px; background: linear-gradient(90deg, var(--brown), var(--brown-dark)); }

  details > summary       { cursor: pointer; color: var(--brown-dark); font-weight: 700; font-size: 14px; margin-top: 10px; }
  details > summary:hover { color: var(--brown-deep); }
  details .sym-list       { font-size: 13px; color: var(--text-light); line-height: 1.8; margin-top: 8px; padding: 10px 14px; background: var(--cream); border-radius: 10px; }

  .treatment-box {
    background: linear-gradient(135deg, #fdf8f5, var(--white));
    border: 2px solid var(--brown-light); border-left: 5px solid var(--brown-dark);
    border-radius: 16px; padding: 18px 22px;
    font-family: 'Nunito', sans-serif; font-size: 15px; color: var(--text); line-height: 1.8;
  }

  .disclaimer {
    background: #fffbe6; border: 2.5px solid #e8b800;
    border-left: 7px solid #c89000; border-radius: 16px;
    padding: 22px 26px; font-size: 17px; font-weight: 700;
    color: #4a3000; font-family: 'Nunito', sans-serif;
    margin-top: 22px; line-height: 1.9;
  }
  .disc-title {
    font-size: 20px; font-weight: 800; color: #3a2000; margin-bottom: 10px;
    display: flex; align-items: center; gap: 8px; font-family: 'Nunito', sans-serif;
  }

  .empty-state { text-align: center; padding: 48px 20px 36px; font-family: 'Nunito', sans-serif; }
  .empty-logo  { font-size: 90px; margin-bottom: 10px; animation: moo 4s ease-in-out infinite; }
  .empty-state h3 { font-family: 'Lilita One', cursive; font-size: 26px; color: var(--brown-dark); margin: 0 0 10px; }
  .empty-state p  { font-size: 15px; color: var(--text-light); margin: 0 0 8px; line-height: 1.7; }
  .sev-legend  { display: flex; justify-content: center; gap: 18px; flex-wrap: wrap; margin-top: 16px; }
  .sev-pill    { padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: 800; font-family: 'Nunito', sans-serif; }

  .sb-brand {
    text-align: center; padding: 22px 0 14px;
    border-bottom: 1px solid rgba(206,144,113,0.35); margin-bottom: 8px;
  }
  .sb-logo    { font-size: 52px; line-height: 1; }
  .sb-name    { font-family: 'Lilita One', cursive; font-size: 22px; color: #fff !important; letter-spacing: 2.5px; margin-top: 4px; }
  .sb-tagline { font-family: 'Lora', serif; font-style: italic; font-size: 14px; color: #e8c4a8 !important; margin-top: 3px; }

  hr { border-color: var(--brown-light) !important; margin: 18px 0 !important; }
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }
  [data-testid="stToolbar"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Load Artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    df           = pd.read_csv("model_ready_dataset.csv")
    matrix       = pd.read_csv("symptom_matrix.csv")
    vectorizer   = joblib.load("tfidf_vectorizer.pkl")
    tfidf_matrix = joblib.load("tfidf_matrix.pkl")
    rf_model     = joblib.load("model_random_forest.pkl")
    le           = joblib.load("label_encoder.pkl")

    def parse_list(val):
        try:
            p = ast.literal_eval(val)
            return p if isinstance(p, list) else []
        except:
            return []

    df["symptoms_list"] = df["symptoms_list"].apply(parse_list)
    all_symptoms = [c for c in matrix.columns if c not in ["disease", "disease_encoded"]]
    return df, matrix, vectorizer, tfidf_matrix, rf_model, le, all_symptoms

df, matrix, vectorizer, tfidf_matrix, rf_model, le, all_symptoms = load_artifacts()

KNOWN_SYMPTOMS = set(all_symptoms)

# ── Constants — synced with feature_engineering.py ────────────────────────────
NOISE_TOKENS = {
    "namely","among others","such as","including","like","etc","and","or",
    "the","a","an","of","in","is","are","with","to","that","which","as",
    "by","at","be","has","have","had","was","were","been","also","this",
    "these","those","its","it","he","she","they","there","their","where",
    "when","if","so","but","on","for","from","not","no","can","will","may",
    "more","yang","dan","atau","dengan","pada","dalam","dari","untuk",
    "tidak","ada","juga","dapat","akan","oleh","sebagai","karena","adalah",
    "serta","bila","jika",
}

INDONESIAN_MARKERS = {
    "dengan","dan","atau","pada","yang","untuk","dari","tidak","ada","ini",
    "itu","juga","dapat","akan","sebagai","karena","oleh","dalam","adalah",
    "serta","bahwa","bila","jika","seperti","antara","setelah",
    "gejala","penyakit","penanganan","risiko","diagnosa","obat","sakit",
    "hewan","ternak","sapi",
}

# Synced with feature_engineering.py — guards the UI from displaying
# any PDF artifact rows that may have slipped into model_ready_dataset.csv
PDF_ARTIFACT_PREFIXES = (
    "tindakan sanitasi",
    "pengaturan yang",
    "pada daerah enzootic",
    "bangkai langsung",
    "yang dilakukan oleh",
    "tidak sakit",
    "pencegahan",
    "penanganan",
    "pengobatan",
)

SEVERITY_WEIGHTS = {
    "seizure":3,"convulsion":3,"unconscious":3,"difficulty breathing":3,
    "respiratory":3,"collapse":3,"paralysis":3,"hemorrhage":3,"bleeding":3,
    "sudden death":3,"coma":3,"shock":3,"high fever":3,"abort":3,"miscarriage":3,
    "fever":2,"diarrhea":2,"vomiting":2,"swelling":2,"lameness":2,
    "staggering":2,"weakness":2,"dehydration":2,"trembling":2,
    "decreased milk":2,"weight loss":2,"anemia":2,"discharge":2,"ulcer":2,"lesion":2,
    "anorexia":1,"loss of appetite":1,"dull fur":1,"itching":1,"hair loss":1,
    "lethargy":1,"constipation":1,"blurred vision":1,"excessive saliva":1,
    "fatigue":1,"anestrus":1,"skin problems":1,
}

SEV_ICON  = {"Critical":"🔴","Severe":"🟠","Moderate":"🟡","Mild":"🟢","Unknown":"⚪"}
SEV_CLASS = {"Critical":"critical","Severe":"severe","Moderate":"moderate","Mild":"mild","Unknown":""}


# ── Helpers ────────────────────────────────────────────────────────────────────

def normalize_input(user_symptoms: str) -> list[str]:
    tokens = [s.strip().lower() for s in user_symptoms.split(",") if s.strip()]
    seen, clean = set(), []
    for t in tokens:
        if t in NOISE_TOKENS or len(t) <= 1: continue
        if re.fullmatch(r'[\d\s\.,]+', t):   continue
        if len(t.split()) > 6:               continue
        if t not in seen:
            clean.append(t); seen.add(t)
    return clean


def has_indonesian(text: str) -> bool:
    return bool(set(str(text).lower().split()) & INDONESIAN_MARKERS)


def is_clean_disease(name: str) -> bool:
    """
    Synced with feature_engineering.py and model_training.py.
    Filters out PDF artifact sentence fragments, Indonesian residue,
    digit-prefixed names, and overly long names.
    """
    name_lower = name.strip().lower()
    if name_lower.startswith(PDF_ARTIFACT_PREFIXES):
        return False
    if len(name.split()) > 6:
        return False
    if any(c.isdigit() for c in name[:5]):
        return False
    if has_indonesian(name):
        return False
    return True


def split_known_unknown(tokens: list[str]) -> tuple[list[str], list[str]]:
    """
    Split tokens into known (in dataset vocabulary) and unknown.
    Uses exact match then partial match for near-misses.
    """
    known, unknown = [], []
    for t in tokens:
        if t in KNOWN_SYMPTOMS:
            known.append(t)
        else:
            partial = any(t in ks or ks in t for ks in KNOWN_SYMPTOMS)
            if partial:
                known.append(t)
            else:
                unknown.append(t)
    return known, unknown


def compute_severity(text: str):
    score = sum(w for kw, w in SEVERITY_WEIGHTS.items() if kw in text.lower())
    count = len(normalize_input(text))
    if score == 0:
        level = "Mild" if count <= 1 else "Moderate" if count <= 3 else "Severe" if count <= 6 else "Critical"
    elif score <= 3:  level = "Mild"
    elif score <= 7:  level = "Moderate"
    elif score <= 12: level = "Severe"
    else:             level = "Critical"
    return level, score


def encode_symptoms(user_symptoms: str) -> np.ndarray:
    tokens = normalize_input(user_symptoms)
    vector = np.zeros(len(all_symptoms))
    for i, s in enumerate(all_symptoms):
        if s in tokens: vector[i] = 1
    return vector.reshape(1, -1)


def diagnose(user_symptoms: str, top_n: int = 3) -> dict:
    tokens       = normalize_input(user_symptoms)
    normalized_q = ", ".join(tokens)
    query_vec    = vectorizer.transform([normalized_q])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    sim_results, seen_d = [], set()
    for idx in np.argsort(similarities)[::-1]:
        if len(sim_results) >= top_n: break
        row     = df.iloc[idx]
        disease = str(row["disease"])
        if not is_clean_disease(disease): continue
        dk = disease.lower()
        if dk in seen_d: continue
        seen_d.add(dk)
        treatment = str(row.get("treatment", "")).strip()
        sim_results.append({
            "disease"         : disease.title(),
            "similarity_score": round(float(similarities[idx]), 4),
            "treatment"       : treatment if treatment and treatment.lower() != "nan" else "No treatment data available.",
            "severity_level"  : str(row.get("severity_level", "Unknown")),
            "matched_symptoms": str(row.get("symptoms", "")),
        })

    sev_level, sev_score = compute_severity(normalized_q)
    return {
        "input"      : user_symptoms.strip(),
        "normalized" : normalized_q,
        "sim_results": sim_results,
        "severity"   : sev_level,
        "sev_score"  : sev_score,
        "top_disease": sim_results[0]["disease"]   if sim_results else "Unknown",
        "treatment"  : sim_results[0]["treatment"] if sim_results else "No data available.",
    }


# ── Session State ──────────────────────────────────────────────────────────────
if "history"       not in st.session_state: st.session_state.history       = []
if "selected_hist" not in st.session_state: st.session_state.selected_hist = None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
      <div class='sb-brand'>
        <div class='sb-logo'>🐄</div>
        <div class='sb-name'>BOVINITY</div>
        <div class='sb-tagline'>infinite care for every cow</div>
      </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🕐 Past Diagnoses")

    if not st.session_state.history:
        st.markdown(
            "<div style='font-size:13px; color:#ddb89a; padding:4px 0 8px;'>"
            "No history yet — run a diagnosis first!</div>",
            unsafe_allow_html=True,
        )
    else:
        for i, h in enumerate(reversed(st.session_state.history)):
            real_idx = len(st.session_state.history) - 1 - i
            sev   = h["severity"]
            icon  = SEV_ICON.get(sev, "⚪")
            label = h["top_disease"]
            short = h["input"][:32] + "…" if len(h["input"]) > 32 else h["input"]

            if st.button(f"{icon}  {label}", key=f"hist_btn_{real_idx}", use_container_width=True):
                st.session_state.selected_hist = real_idx
                st.rerun()

            st.markdown(
                f"<div style='font-size:11px; color:#ddb89a; margin:-8px 0 6px 4px;'>{short}</div>",
                unsafe_allow_html=True,
            )

        if st.button("🗑️  Clear All History", use_container_width=True):
            st.session_state.history       = []
            st.session_state.selected_hist = None
            st.rerun()

    st.markdown("---")
    st.markdown("""
      <div style='font-size:20px; color:#ddb89a; font-family:Nunito,sans-serif; line-height:1.7; font-weight:700;'>
        ⚠️ For <b style="color:#fde8d8;">preliminary guidance only.</b><br>
        Always consult a licensed veterinarian before giving any treatment.
      </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
  <div class='hero-outer'>
    <div class='hero-left'>
      <div class='hero-title'>BOVINITY</div>
      <div class='hero-tagline'>infinite care for every cow</div>
    </div>
    <div class='hero-logo'>🐄</div>
  </div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

with st.expander("👋  New here? How to use Bovinity — tap to read", expanded=False):
    st.markdown("""
    <div class='howto-box'>
      <div class='howto-step'><div class='step-num'>1</div>
        <div>Look at your cow carefully and note what looks wrong — for example: <em>not eating, has a fever, runny nose, limping</em></div></div>
      <div class='howto-step'><div class='step-num'>2</div>
        <div>Type those symptoms in the box below, <b>separated by commas</b></div></div>
      <div class='howto-step'><div class='step-num'>3</div>
        <div>Press the big <b>🔍 Diagnose</b> button and wait a moment</div></div>
      <div class='howto-step'><div class='step-num'>4</div>
        <div>Read the results — Bovinity will show the likely illness, how serious it is, and what treatment may help</div></div>
      <div style='background:#fff3cd; border-radius:10px; padding:10px 14px; margin-top:6px; font-size:14px; color:#7a5c00;'>
        ⚠️ <b>Always show results to a real veterinarian before giving any medicine.</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="sec-label">💬 What symptoms does the cow have?</div>', unsafe_allow_html=True)

col_input, col_btn = st.columns([4, 1])
with col_input:
    typed = st.text_input(
        "symptom_input",
        placeholder="e.g.  fever, not eating, runny nose, weak legs …",
        label_visibility="collapsed",
        key="main_input",
    )
with col_btn:
    go = st.button("🔍  Diagnose", use_container_width=True, type="primary")

# ── Chip preview + unknown symptom warning ────────────────────────────────────
if typed.strip():
    tokens         = normalize_input(typed)
    known, unknown = split_known_unknown(tokens)
    unknown_set    = set(unknown)

    if tokens:
        chips_html = ""
        for t in tokens:
            chip_cls   = "chip-unknown" if t in unknown_set else "chip"
            chips_html += f'<span class="{chip_cls}">🩺 {t.capitalize()}</span>'
        st.markdown(f'<div class="chips-wrap">{chips_html}</div>', unsafe_allow_html=True)

    if unknown:
        unknown_list = ", ".join(f"<b>{u}</b>" for u in unknown)
        st.markdown(f"""
          <div class='unknown-warn'>
            ⚠️ The following symptom(s) were <b>not found in our dataset</b>:
            {unknown_list}<br><br>
            These will be ignored during diagnosis. Please try rephrasing them
            using simpler terms — for example, <em>"high temperature"</em> → <em>"fever"</em>,
            or <em>"won't walk"</em> → <em>"lameness"</em>.
          </div>
        """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Run diagnosis ──────────────────────────────────────────────────────────────
if go:
    if not typed.strip():
        st.warning("⚠️  Please type at least one symptom to continue.")
    else:
        tokens        = normalize_input(typed)
        known, unknown = split_known_unknown(tokens)

        if not known:
            st.markdown("""
              <div class='unknown-warn'>
                ❌ <b>None of the symptoms you entered were recognised in our dataset.</b><br><br>
                Please try describing the symptoms using simpler, common veterinary terms —
                for example: <em>fever, diarrhea, weakness, lameness, hair loss, anorexia</em>.
              </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("🐄  Bovinity is thinking…"):
                result = diagnose(typed)
            st.session_state.history.append(result)
            st.session_state.selected_hist = len(st.session_state.history) - 1
            st.rerun()

display_result = None
if st.session_state.selected_hist is not None and st.session_state.history:
    idx = st.session_state.selected_hist
    if 0 <= idx < len(st.session_state.history):
        display_result = st.session_state.history[idx]


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════
if display_result:
    r   = display_result
    sev = r["severity"]

    st.markdown(f'<div class="bubble-user">🧑‍🌾 &nbsp; {r["input"].title()}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="bubble-bot">🐄 &nbsp; I found <b>{len(r["sim_results"])}</b> possible condition(s) '
        f'matching these symptoms.<br>Overall severity: '
        f'<span class="sev-badge sev-{sev}">{SEV_ICON.get(sev,"⚪")} {sev}</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec-label">🔬 Most Likely Conditions</div>', unsafe_allow_html=True)
    for i, match in enumerate(r["sim_results"]):
        pct      = int(match["similarity_score"] * 100)
        sev_m    = match.get("severity_level", "Unknown")
        card_cls = SEV_CLASS.get(sev_m, "")
        icon_m   = SEV_ICON.get(sev_m, "⚪")
        sym_text = match["matched_symptoms"]
        st.markdown(f"""
        <div class='result-card {card_cls}'>
          <div class='card-row'>
            <div>
              <div class='card-title'>#{i+1} &nbsp; {match['disease']}</div>
              <div style='margin-top:6px;'><span class='sev-badge sev-{sev_m}'>{icon_m} {sev_m}</span></div>
            </div>
            <div class='card-pct'>{pct}%</div>
          </div>
          <div class='bar-wrap'>
            <div class='bar-label'>SYMPTOM MATCH</div>
            <div class='bar-bg'><div class='bar-fg' style='width:{pct}%;'></div></div>
          </div>
          <details>
            <summary>📋 View known symptoms for this disease</summary>
            <div class='sym-list'>{sym_text if sym_text and sym_text.lower() != 'nan' else 'No symptom data available.'}</div>
          </details>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label">💊 Suggested Treatment Plan</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="treatment-box">🩺 &nbsp; {r["treatment"]}</div>', unsafe_allow_html=True)

    st.markdown("""
      <div class='disclaimer'>
        <div class='disc-title'>⚠️ Important Disclaimer</div>
        Bovinity provides <b>preliminary guidance only</b> based on a limited dataset.
        Always consult a <b>licensed veterinarian</b> before administering any treatment
        to your animal.<br><br>
        If the severity level is 🔴 <b>Critical</b> or 🟠 <b>Severe</b>,
        please seek veterinary help <b>immediately.</b>
      </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# EMPTY STATE
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("""
      <div class='empty-state'>
        <div class='empty-logo'>🐄</div>
        <h3>Hello! I'm Bovinity 👋</h3>
        <p>
          Type your cow's symptoms in the box above — for example:<br>
          <em>"fever, not eating, runny nose, weak legs"</em><br><br>
          Then press <b>🔍 Diagnose</b> to get started.
        </p>
        <div class='sev-legend'>
          <span class='sev-pill' style='background:#e0f8ec; color:#1a7a4a;'>🟢 Mild</span>
          <span class='sev-pill' style='background:#fff8d4; color:#a07800;'>🟡 Moderate</span>
          <span class='sev-pill' style='background:#ffebd4; color:#d4600a;'>🟠 Severe</span>
          <span class='sev-pill' style='background:#fde8e8; color:#c0392b;'>🔴 Critical</span>
        </div>
      </div>
    """, unsafe_allow_html=True)

# py -m streamlit run "c:/Users/Anusree/Desktop/College Things/S4/Data Science/Data science project/chatbot interface.py"