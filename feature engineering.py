import pandas as pd
import ast
import re
import json
import logging
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
COMBINED_PATH      = Path("cleaned_dataset_combined.csv")
OUTPUT_PATH        = Path("featured_dataset.csv")
MATRIX_PATH        = Path("symptom_matrix.csv")
LABEL_ENCODER_PATH = Path("label_encoder.pkl")
TFIDF_PATH         = Path("tfidf_vectorizer.pkl")
TFIDF_MATRIX_PATH  = Path("tfidf_matrix.pkl")
SEVERITY_PATH      = Path("severity_rules.json")

# ── Noise tokens ──────────────────────────────────────────────────────────────
NOISE_TOKENS = {
    "namely", "among others", "such as", "including", "like",
    "etc", "and", "or", "the", "a", "an", "of", "in", "is",
    "are", "with", "to", "that", "which", "as", "by", "at",
    "be", "has", "have", "had", "was", "were", "been", "also",
    "this", "these", "those", "its", "it", "he", "she", "they",
    "there", "their", "where", "when", "if", "so", "but", "on",
    "for", "from", "not", "no", "can", "will", "may", "more",
    "disease", "disease_encoded",
    "yang", "dan", "atau", "dengan", "pada", "dalam", "dari",
    "untuk", "tidak", "ada", "juga", "dapat", "akan", "oleh",
    "sebagai", "karena", "adalah", "serta", "bila", "jika",
}

# ── Indonesian markers ────────────────────────────────────────────────────────
INDONESIAN_MARKERS = {
    "dengan", "dan", "atau", "pada", "yang", "untuk", "dari",
    "tidak", "ada", "ini", "itu", "juga", "dapat", "akan",
    "sebagai", "karena", "oleh", "dalam", "adalah", "serta",
    "bahwa", "bila", "jika", "seperti", "antara", "setelah",
}

# ── PDF artifact disease name blocklist ───────────────────────────────────────
# These are sentence fragments that leaked from the PDF parser and got picked
# up as disease names. They are NOT diseases — they are treatment/prevention
# text that the block parser grabbed by mistake.
# Identified from training output: rows 49-53 in the flagged disease list.
# Uses startswith matching (lowercase) for robustness.
PDF_ARTIFACT_PREFIXES = (
    "tindakan sanitasi",       # "sanitation measures that can be done..."
    "pengaturan yang",         # "strict regulations on..."
    "pada daerah enzootic",    # "in enzootic areas, with vaccination..."
    "bangkai langsung",        # "carcasses directly burned or buried..."
    "yang dilakukan oleh",     # "done by humans is to avoid..."
    "tidak sakit",             # "not sick" — not a disease name
    "pencegahan",              # prevention text leaked as disease
    "penanganan",              # treatment text leaked as disease
    "pengobatan",              # medication text leaked as disease
)

# ── Manual disease name translations ─────────────────────────────────────────
# Covers all Indonesian disease names identified from the training output,
# including the new ones flagged in the 30-row WARNING.
DISEASE_TRANSLATIONS = {
    # ── Original entries ──────────────────────────────────────────────────────
    "abses"                        : "abscess",
    "corpus luteum persisten"      : "persistent corpus luteum",
    "distokia"                     : "dystocia",
    "endometritis"                 : "endometritis",
    "hipokalsemia"                 : "hypocalcemia",
    "kembung"                      : "bloat",
    "keracunan"                    : "poisoning",
    "keracunan tanaman"            : "plant poisoning",
    "ketosis"                      : "ketosis",
    "laminitis"                    : "laminitis",
    "mastitis"                     : "mastitis",
    "metritis"                     : "metritis",
    "penyakit jembrana"            : "jembrana disease",
    "penyakit mulut dan kuku"      : "foot and mouth disease",
    "penyakit ngorok"              : "bovine respiratory disease",
    "penyakit surra"               : "surra",
    "pink eye"                     : "pink eye",
    "pneumonia"                    : "pneumonia",
    "radang paru-paru"             : "pneumonia",
    "radang usus"                  : "enteritis",
    "retensi plasenta"             : "retained placenta",
    "salmonellosis"                : "salmonellosis",
    "scabies"                      : "scabies",
    "selulitis"                    : "cellulitis",
    "tympani"                      : "tympany",
    "trypanosomiasis"              : "trypanosomiasis",
    "vibriosis"                    : "vibriosis",
    "cacingan"                     : "helminthiasis",
    "kudis"                        : "mange",
    "kuku busuk"                   : "foot rot",
    "kurap"                        : "ringworm",
    "orf"                          : "orf",
    "pincang"                      : "lameness",
    "prolaps"                      : "prolapse",
    "prolaps uterus"               : "uterine prolapse",
    "prolaps vagina"               : "vaginal prolapse",
    # ── New entries from training output warnings ─────────────────────────────
    "malnutrisi"                   : "malnutrition",
    "hipofungsi ovari"             : "ovarian hypofunction",
    "sapi ingusan"                 : "mucosal disease",
    "sapi surra"                   : "surra",
    "sapi demam bovine ephemeral"  : "bovine ephemeral fever",
    # Pink eye row had extra garbage text appended — map the core name
    "pink eye penyakit mata"       : "pink eye",
    # Foot rot row had extra PDF text appended
    "foot rot"                     : "foot rot",
    # Fasciolosis / liver fluke
    "fasciolosis"                  : "fasciolosis",
    "fasciola"                     : "fasciolosis",
    # Q fever
    "q fever"                      : "q fever",
    # Silent heat (reproductive)
    "silent heat"                  : "silent heat",
    "silent oestrus"               : "silent heat",
    # Babesiosis / tick fever
    "babesiosis"                   : "babesiosis",
    "piroplasmosis"                : "babesiosis",
    # Avitaminosis
    "avitaminosis"                 : "avitaminosis",
    # Brucellosis
    "brucellosis"                  : "brucellosis",
    # IBR
    "infectious bovine rhinotracheitis" : "infectious bovine rhinotracheitis",
    "ibr"                          : "infectious bovine rhinotracheitis",
    # BVD
    "bovine viral diarrhea"        : "bovine viral diarrhea",
    "bvd"                          : "bovine viral diarrhea",
    # Theileriosis
    "theileriosis"                 : "theileriosis",
    # Anaplasmosis
    "anaplasmosis"                 : "anaplasmosis",
    # Dermatophilosis
    "dermatophilosis"              : "dermatophilosis",
    # Actinomycosis
    "actinomycosis"                : "actinomycosis",
    # Wooden tongue
    "actinobacillosis"             : "wooden tongue",
    "wooden tongue"                : "wooden tongue",
    # Hypocalcemia / milk fever
    "milk fever"                   : "milk fever",
    "parturient paresis"           : "milk fever",
    # Hardware disease
    "traumatic reticuloperitonitis": "hardware disease",
    "hardware disease"             : "hardware disease",
}

# ── Symptom prefix artifacts from MarianMT ───────────────────────────────────
STRIP_PREFIXES = re.compile(
    r'^(and |or |the |a |an |also |with |has |have |there is |there are )',
    flags=re.IGNORECASE,
)

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


# ── Shared helpers ────────────────────────────────────────────────────────────

def parse_symptoms_list(val) -> list:
    try:
        parsed = ast.literal_eval(val)
        return parsed if isinstance(parsed, list) else []
    except:
        return []


def has_indonesian(text: str) -> bool:
    words = set(str(text).lower().split())
    return bool(words & INDONESIAN_MARKERS)


def is_pdf_artifact(name: str) -> bool:
    """
    Returns True if the disease name is a PDF extraction artifact —
    i.e. a sentence fragment from treatment/prevention text that leaked
    through the block parser instead of being a real disease name.
    Identified from rows 49-53 in the training output warning list.
    """
    name_lower = name.strip().lower()
    return name_lower.startswith(PDF_ARTIFACT_PREFIXES)


def translate_disease_name(name: str) -> str:
    """
    Translate a disease name that MarianMT failed to translate.
    1. Checks exact match in DISEASE_TRANSLATIONS (case-insensitive)
    2. Checks if any known key is a substring of the name
       (handles PDF artifact names like "Pink Eye Penyakit Mata A Mata Berair..."
        which start with a known key)
    3. Falls back to the original name and logs a warning if still Indonesian
    """
    name_stripped = name.strip()
    name_lower    = name_stripped.lower()

    # Exact match
    if name_lower in DISEASE_TRANSLATIONS:
        return DISEASE_TRANSLATIONS[name_lower]

    # Partial / startswith match — handles PDF garbage appended to real names
    for key, translation in DISEASE_TRANSLATIONS.items():
        if name_lower.startswith(key):
            return translation

    # No match found
    if has_indonesian(name_stripped):
        logger.warning(
            f"Disease name '{name_stripped}' still contains Indonesian text "
            f"and has no entry in DISEASE_TRANSLATIONS. Add it manually to fix."
        )
    return name_stripped


def clean_disease_column(series: pd.Series) -> pd.Series:
    """
    Apply translate_disease_name to every row in the disease column.
    Strips whitespace and normalises to title case.
    """
    return series.apply(lambda x: translate_disease_name(str(x).strip()).title())


def strip_marian_prefix(token: str) -> str:
    return STRIP_PREFIXES.sub("", token).strip()


def is_clean_symptom(s: str) -> bool:
    """
    Returns True if the symptom token is valid after prefix-stripping.
    """
    s = strip_marian_prefix(s.strip().lower())
    if not s:                                  return False
    if s in NOISE_TOKENS:                      return False
    if len(s) <= 1:                            return False
    if re.fullmatch(r'[\d\s\.,]+', s):         return False
    if len(s.split()) > 6:                     return False
    if s.endswith("."):                        return False
    words = s.split()
    if len(words) >= 2 and len(set(words)) == 1: return False
    if all(w in INDONESIAN_MARKERS for w in words): return False
    return True


def clean_symptom_tokens(symptoms: list) -> list:
    seen    = set()
    cleaned = []
    for s in symptoms:
        s = strip_marian_prefix(s.strip().lower())
        if not is_clean_symptom(s):
            continue
        if s not in seen:
            cleaned.append(s)
            seen.add(s)
    return cleaned


# ── STEP 1 — Clean, Translate & Filter ────────────────────────────────────────
logger.info("── Step 1: Clean, Translate, Filter PDF Artifacts & Deduplicate ──")

if not COMBINED_PATH.exists():
    raise FileNotFoundError(
        f"Combined dataset not found: {COMBINED_PATH}. Run preprocessing first."
    )

df = pd.read_csv(COMBINED_PATH)
df["symptoms_list"] = df["symptoms_list"].apply(parse_symptoms_list)
logger.info(f"Loaded {len(df)} rows.")

# ── 1a. Remove PDF artifact rows BEFORE translation ───────────────────────────
before_artifact = len(df)
df = df[~df["disease"].apply(is_pdf_artifact)].reset_index(drop=True)
removed_artifacts = before_artifact - len(df)
if removed_artifacts > 0:
    logger.info(f"Removed {removed_artifacts} PDF artifact rows (sentence fragments mistaken as disease names).")
else:
    logger.info("No PDF artifact rows detected.")

# ── 1b. Translate untranslated disease names ──────────────────────────────────
logger.info("Translating Indonesian disease names...")
df["disease"] = clean_disease_column(df["disease"])

# Re-check after translation
for col in ["symptoms", "disease"]:
    flagged = df[col].apply(has_indonesian).sum()
    if flagged > 0:
        logger.warning(
            f"Column '{col}': {flagged} rows still contain Indonesian text. "
            f"Add missing entries to DISEASE_TRANSLATIONS and re-run."
        )
        print(f"\n── Remaining Indonesian in '{col}' ──")
        print(df[df[col].apply(has_indonesian)][["disease", col]].to_string())
    else:
        logger.info(f"Column '{col}': ✓ No Indonesian markers found.")

# ── 1c. Clean symptom tokens ──────────────────────────────────────────────────
df["symptoms_list"] = df["symptoms_list"].apply(clean_symptom_tokens)
df["symptoms"]      = df["symptoms_list"].apply(lambda x: ", ".join(x))

# Drop rows emptied by cleaning
before = len(df)
df = df[df["symptoms_list"].apply(len) > 0].reset_index(drop=True)
logger.info(f"Dropped {before - len(df)} rows with empty symptoms after cleaning.")

# Deduplicate
before = len(df)
df = df.drop_duplicates(subset=["disease", "symptoms"]).reset_index(drop=True)
logger.info(f"Dropped {before - len(df)} duplicate rows. Remaining: {len(df)}")

all_syms = [s for sl in df["symptoms_list"] for s in sl]
logger.info(f"Total unique symptoms after cleaning: {len(set(all_syms))}")

df.to_csv(OUTPUT_PATH, index=False)
logger.info(f"Saved → '{OUTPUT_PATH}'")
print("\n── Sample after Step 1 ──")
print(df[["disease", "symptoms"]].head(10).to_string())


# ── STEP 2 — Build Binary Symptom Matrix ──────────────────────────────────────
logger.info("── Step 2: Build Binary Symptom Matrix ──")

df = pd.read_csv(OUTPUT_PATH)
df["symptoms_list"] = df["symptoms_list"].apply(parse_symptoms_list)

RESERVED_COLS = {"disease", "disease_encoded"}
all_symptoms  = sorted(set(
    s for sl in df["symptoms_list"] for s in sl
    if s not in RESERVED_COLS
    and s not in NOISE_TOKENS
    and len(s) > 1
))

logger.info(f"Building binary matrix: {len(df)} rows × {len(all_symptoms)} symptom columns...")

matrix = pd.DataFrame(0, index=df.index, columns=all_symptoms)
for i, symptoms in enumerate(df["symptoms_list"]):
    for s in symptoms:
        if s in matrix.columns:
            matrix.at[i, s] = 1

matrix.insert(0, "disease", df["disease"].values)

le = LabelEncoder()
matrix["disease_encoded"] = le.fit_transform(matrix["disease"])

logger.info(f"Unique disease classes : {len(le.classes_)}")
logger.info(f"Matrix shape           : {matrix.shape}")

feature_cols = matrix.drop(columns=["disease", "disease_encoded"])
sparsity     = 1 - (feature_cols.values.sum() / feature_cols.size)
logger.info(f"Matrix sparsity        : {sparsity*100:.1f}%")

matrix.to_csv(MATRIX_PATH, index=False)
joblib.dump(le, LABEL_ENCODER_PATH)
logger.info(f"Saved binary matrix   → '{MATRIX_PATH}'")
logger.info(f"Saved label encoder   → '{LABEL_ENCODER_PATH}'")

print("\n── Matrix preview (first 5 symptom columns) ──")
print(matrix[["disease", "disease_encoded"]].head(10).to_string())
print(matrix.iloc[:5, 1:6].to_string())


# ── STEP 3 — Build TF-IDF Matrix ──────────────────────────────────────────────
logger.info("── Step 3: Build TF-IDF Matrix ──")

df = pd.read_csv(OUTPUT_PATH)
logger.info(f"Loaded {len(df)} rows.")

vectorizer = TfidfVectorizer(
    analyzer     = "word",
    ngram_range  = (1, 2),
    min_df       = 1,
    max_df       = 1.0,
    sublinear_tf = True,
)

tfidf_matrix = vectorizer.fit_transform(df["symptoms"])
logger.info(f"TF-IDF matrix shape : {tfidf_matrix.shape}")
logger.info(f"Vocabulary size     : {len(vectorizer.vocabulary_)}")

test_queries = [
    "fever diarrhea weakness",
    "hair loss itching skin problems",
    "anorexia anestrus",
]

print("\n── Sanity Check: Similarity Matcher ──")
for test_query in test_queries:
    query_vec    = vectorizer.transform([test_query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices  = np.argsort(similarities)[::-1][:3]
    print(f"\n  Query: '{test_query}'")
    for idx in top_indices:
        print(f"    Disease  : {df['disease'].iloc[idx]}")
        print(f"    Score    : {similarities[idx]:.4f}")
        print(f"    Symptoms : {df['symptoms'].iloc[idx][:80]}...")

joblib.dump(vectorizer,   TFIDF_PATH)
joblib.dump(tfidf_matrix, TFIDF_MATRIX_PATH)
logger.info(f"Saved TF-IDF vectorizer → '{TFIDF_PATH}'")
logger.info(f"Saved TF-IDF matrix     → '{TFIDF_MATRIX_PATH}'")


# ── STEP 4 — Build Severity Scorer ────────────────────────────────────────────
logger.info("── Step 4: Build Severity Scorer ──")

df = pd.read_csv(OUTPUT_PATH)

def score_severity(symptoms_text: str) -> dict:
    text    = symptoms_text.lower()
    score   = 0
    matched = []

    for keyword, weight in SEVERITY_WEIGHTS.items():
        if keyword in text:
            score += weight
            matched.append((keyword, weight))

    clean_count = len([s.strip() for s in symptoms_text.split(",") if s.strip()])

    if score == 0:
        if clean_count <= 1:   level = "Mild"
        elif clean_count <= 3: level = "Moderate"
        elif clean_count <= 6: level = "Severe"
        else:                  level = "Critical"
    elif score <= 3:  level = "Mild"
    elif score <= 7:  level = "Moderate"
    elif score <= 12: level = "Severe"
    else:             level = "Critical"

    return {
        "severity_score"  : score,
        "severity_level"  : level,
        "matched_keywords": matched,
    }

severity_results     = df["symptoms"].apply(score_severity)
df["severity_score"] = severity_results.apply(lambda x: x["severity_score"])
df["severity_level"] = severity_results.apply(lambda x: x["severity_level"])

print("\n── Severity Distribution ──")
print(df["severity_level"].value_counts().to_string())

unknown = df[df["severity_level"] == "Unknown"]
if len(unknown) > 0:
    logger.warning(f"{len(unknown)} diseases scored 'Unknown' — check their symptoms.")
    print(unknown[["disease", "symptoms"]].to_string())
else:
    logger.info("✓ All diseases have a severity level assigned.")

print("\n── Sample Severity Scores ──")
print(df[["disease", "severity_score", "severity_level"]].head(10).to_string())

with open(SEVERITY_PATH, "w") as f:
    json.dump(SEVERITY_WEIGHTS, f, indent=2)

df.to_csv(OUTPUT_PATH, index=False)
logger.info(f"Saved severity rules     → '{SEVERITY_PATH}'")
logger.info(f"Updated featured dataset → '{OUTPUT_PATH}'")


# ── STEP 5 — Verify All Feature Files ─────────────────────────────────────────
logger.info("── Step 5: File Verification ──")

files = {
    "Featured dataset" : Path("featured_dataset.csv"),
    "Symptom matrix"   : Path("symptom_matrix.csv"),
    "TF-IDF vectorizer": Path("tfidf_vectorizer.pkl"),
    "TF-IDF matrix"    : Path("tfidf_matrix.pkl"),
    "Label encoder"    : Path("label_encoder.pkl"),
    "Severity rules"   : Path("severity_rules.json"),
}

print("\n════════════════════════════════════")
print("   FEATURE ENGINEERING FILE CHECK   ")
print("════════════════════════════════════")

all_ok = True
for name, path in files.items():
    exists = path.exists()
    status = "✓" if exists else "✗ MISSING"
    size   = f"{path.stat().st_size / 1024:.1f} KB" if exists else ""
    print(f"  {status}  {name:<22} {size}")
    if not exists:
        all_ok = False

print("════════════════════════════════════")

if all_ok:
    df     = pd.read_csv("featured_dataset.csv")
    matrix = pd.read_csv("symptom_matrix.csv")
    le     = joblib.load("label_encoder.pkl")

    indonesian_symptoms = df["symptoms"].apply(has_indonesian).sum()
    indonesian_diseases = df["disease"].apply(has_indonesian).sum()

    feature_cols      = [c for c in matrix.columns if c not in ["disease", "disease_encoded"]]
    disease_col_leak  = "disease" in feature_cols or "disease_encoded" in feature_cols

    print(f"\n  Rows in featured dataset  : {len(df)}")
    print(f"  Symptom matrix shape      : {matrix.shape}")
    print(f"  Symptom feature columns   : {len(feature_cols)}")
    print(f"  Disease classes           : {len(le.classes_)}")
    print(f"  Severity levels present   : {df['severity_level'].value_counts().to_dict()}")
    print(f"  Symptoms w/ Indonesian    : {indonesian_symptoms} "
          f"{'⚠️  re-check translation' if indonesian_symptoms > 0 else '✓'}")
    print(f"  Diseases w/ Indonesian    : {indonesian_diseases} "
          f"{'⚠️  add to DISEASE_TRANSLATIONS' if indonesian_diseases > 0 else '✓'}")
    print(f"  'disease' column leak     : {'⚠️  PRESENT — fix needed' if disease_col_leak else '✓ Not present'}")

    if indonesian_diseases > 0:
        print(f"\n── Untranslated disease names (add to DISEASE_TRANSLATIONS) ──")
        print(df[df["disease"].apply(has_indonesian)]["disease"].unique())

    print("\n  ✓ All files ready. Proceed to Model Training.")
else:
    print("\n  ✗ Some files are missing. Re-run the steps above in order.")