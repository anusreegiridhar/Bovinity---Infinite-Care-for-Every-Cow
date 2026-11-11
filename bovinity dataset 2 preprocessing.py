import pdfplumber
import re
import pandas as pd
import logging
from pathlib import Path
from transformers import MarianMTModel, MarianTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Constants
INPUT_PDF     = Path(r"C:\Users\Anusree\Desktop\College Things\S4\Data Science\Data science project\dataset 2.pdf")
OUTPUT_PATH   = Path("cleaned_dataset2.csv")
DATASET1_PATH = Path("cleaned_dataset1.csv")
COMBINED_PATH = Path("cleaned_dataset_combined.csv")
MODEL_NAME    = "Helsinki-NLP/opus-mt-id-en"
MAX_SYMPTOMS  = 5
BATCH_SIZE    = 16
MAX_CHUNK_LEN = 400
SKIP_KEYWORDS = {"solusi", "pencegahan", "pengobatan", "komplikasi", "penyebab"}
NORMALIZE_COLS= ["symptoms", "disease", "treatment"]

INDONESIAN_MARKERS = {
    "dengan", "dan", "atau", "pada", "yang", "untuk", "dari",
    "tidak", "ada", "ini", "itu", "juga", "dapat", "akan",
    "sebagai", "karena", "oleh", "dalam", "adalah", "serta",
    "bahwa", "bila", "jika", "seperti", "antara", "setelah"
}

# Load MarianMT
logger.info("Loading MarianMT Indonesian→English model...")
logger.info("(First run downloads ~300MB — cached instantly after)")
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model     = MarianMTModel.from_pretrained(MODEL_NAME)
logger.info("Model ready.")


# Translation helpers
def translate_batch(texts: list[str]) -> list[str]:
    """Translate a batch of strings. Returns originals on failure."""
    if not texts:
        return []
    try:
        inputs  = tokenizer(
            texts,
            return_tensors = "pt",
            padding        = True,
            truncation     = True,
            max_length     = 512,
        )
        outputs    = model.generate(**inputs, num_beams=4, early_stopping=True)
        translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return translated
    except Exception as e:
        logger.warning(f"Batch failed: {e}. Returning originals.")
        return texts


def should_skip(text: str) -> bool:
    """Skip empty, very short, or non-linguistic strings (dosage codes etc.)."""
    if not text or not text.strip():
        return True
    if len(text.split()) <= 2:
        return True
    # Skip if mostly numbers/symbols (drug dosages, CF scores from PDF)
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.4:
        return True
    return False


def split_long_text(text: str) -> list[str]:
    """Split on sentence boundaries to stay within MarianMT token limit."""
    if len(text) <= MAX_CHUNK_LEN:
        return [text]
    parts = [p.strip() for p in re.split(r'[.;]', text) if p.strip()]
    return parts if parts else [text]


def translate_column(series: pd.Series) -> pd.Series:
    """Batch-translate an entire Series efficiently."""
    results      = [""] * len(series)
    to_translate = []

    for i, text in enumerate(series):
        text = str(text).strip() if not pd.isna(text) else ""
        if should_skip(text):
            results[i] = text
            continue
        chunks = split_long_text(text)
        to_translate.append((i, chunks))

    if not to_translate:
        return pd.Series(results, index=series.index)

    # Flatten all chunks for batch processing
    flat_texts = []
    flat_map   = []
    for orig_idx, chunks in to_translate:
        for chunk in chunks:
            flat_texts.append(chunk)
            flat_map.append(orig_idx)

    logger.info(f"  Translating {len(flat_texts)} chunks in batches of {BATCH_SIZE}...")

    flat_translated = []
    for start in range(0, len(flat_texts), BATCH_SIZE):
        batch      = flat_texts[start:start + BATCH_SIZE]
        translated = translate_batch(batch)
        flat_translated.extend(translated)
        done = min(start + BATCH_SIZE, len(flat_texts))
        logger.info(f"  Progress: {done}/{len(flat_texts)} chunks")

    # Reconstruct rows from chunks
    chunk_cursor = 0
    for orig_idx, chunks in to_translate:
        n         = len(chunks)
        row_parts = flat_translated[chunk_cursor:chunk_cursor + n]
        results[orig_idx] = ". ".join(row_parts)
        chunk_cursor += n

    return pd.Series(results, index=series.index)


def has_indonesian(text: str) -> bool:
    """Check if text still contains Indonesian words after translation."""
    words = set(str(text).lower().split())
    return bool(words & INDONESIAN_MARKERS)


# Step 1: Extract text from PDF
logger.info("Extracting text from PDF...")

if not INPUT_PDF.exists():
    raise FileNotFoundError(f"PDF not found: {INPUT_PDF}")

pages_text = []
with pdfplumber.open(INPUT_PDF) as pdf:
    logger.info(f"PDF has {len(pdf.pages)} pages.")
    for i, page in enumerate(pdf.pages, start=1):
        # Try extracting as table first (better for structured PDFs)
        tables = page.extract_tables()
        if tables:
            for table in tables:
                for row in table:
                    line = " ".join(str(cell) for cell in row if cell)
                    if line.strip():
                        pages_text.append(line.strip())
        else:
            # Fall back to plain text extraction
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)
            else:
                logger.warning(f"Page {i} returned no text (possibly scanned image).")

text = "\n".join(pages_text)
logger.info(f"Extracted {len(text)} characters total.")
print(text[:1000])  # preview


# Step 2: Split text into disease sections
logger.info("Splitting into disease sections...")

disease_blocks = re.split(r'\n\s*\d+\.\s+', text)
disease_blocks = [block.strip() for block in disease_blocks if block.strip()]
logger.info(f"Total disease sections found: {len(disease_blocks)}")


# Step 3: Extract disease name and symptoms
def extract_block(block: str) -> dict | None:
    """
    Parse a single disease block.
    - First non-empty line → disease name
    - Remaining descriptive lines → symptoms
    - Filters out PDF artifacts: CF/MB/MD scores, headers, sub-numbered lines
    """
    lines = [l.strip() for l in block.split("\n") if l.strip()]
    if not lines:
        return None

    disease_name = lines[0]

    # Skip blocks whose "disease name" is clearly a PDF artifact
    if re.search(r'\d+\.\d+', disease_name):
        return None

    symptoms = []
    for line in lines[1:]:
        is_long_enough   = len(line) > 20
        is_not_header    = not any(kw in line.lower() for kw in SKIP_KEYWORDS)
        is_not_numbered  = not re.match(r'^\d+[\.\)]\s', line)
        is_not_cf_score  = not re.search(r'\b(MB|MD|CF)\b', line)   # PDF score columns
        is_not_separator = not re.match(r'^[-=]{3,}', line)
        is_not_all_caps  = not (line.isupper() and len(line.split()) <= 4)  # table headers
        has_enough_words = len(line.split()) >= 4

        if all([is_long_enough, is_not_header, is_not_numbered,
                is_not_cf_score, is_not_separator, is_not_all_caps,
                has_enough_words]):
            # Strip trailing score patterns like "0,8 0.2 0.6"
            cleaned = re.sub(r'\s+[\d,\.]+\s+[\d,\.]+\s+[\d,\.]+\s*$', '', line).strip()
            if cleaned:
                symptoms.append(cleaned)

    if not symptoms:
        return None

    return {
        "disease"  : disease_name,
        "symptoms" : ", ".join(symptoms[:MAX_SYMPTOMS]),
        "treatment": "",
        "risk"     : "",
    }

data = [r for block in disease_blocks if (r := extract_block(block))]
logger.info(f"Successfully parsed {len(data)} disease records.")


# Step 4: Convert to DataFrame
df2 = pd.DataFrame(data)
print(df2.head())


# Step 5: Translate using MarianMT
logger.info("Translating symptoms with MarianMT (offline, no rate limits)...")
df2["symptoms"] = translate_column(df2["symptoms"])
logger.info("Translation complete.")


# Step 6: Normalize text
def normalize_text(text) -> str:
    text = str(text).lower().strip()
    text = re.sub(r'[^a-z0-9, ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

for col in NORMALIZE_COLS:
    df2[col] = df2[col].apply(normalize_text)


# Step 7: Create symptoms list (AFTER normalization) 
def clean_symptoms(text) -> list[str]:
    if pd.isna(text):
        return []
    symptoms = [s.strip() for s in str(text).split(',') if s.strip()]
    seen     = set()
    return [s for s in symptoms if not (s in seen or seen.add(s))]

df2["symptoms_list"] = df2["symptoms"].apply(clean_symptoms)


# Step 8: Remove empty/invalid rows
before = len(df2)
df2    = df2[df2["symptoms"].str.strip() != ""]
df2    = df2[df2["disease"].str.strip()  != ""]
logger.info(f"Dropped {before - len(df2)} empty rows. Remaining: {len(df2)}")


# Step 9: Quality check — flag remaining Indonesian text
for col in ["symptoms", "disease"]:
    flagged = df2[col].apply(has_indonesian).sum()
    if flagged > 0:
        logger.warning(f"Column '{col}': {flagged} rows may still contain Indonesian text.")
        # Show which rows are flagged so you can inspect them
        flagged_rows = df2[df2[col].apply(has_indonesian)][["disease", col]]
        print(f"\n── Flagged rows in '{col}' ──")
        print(flagged_rows.to_string())
    else:
        logger.info(f"Column '{col}': ✓ Clean.")


# Step 10: Save cleaned dataset 2
df2.to_csv(OUTPUT_PATH, index=False)
logger.info(f"Saved cleaned dataset 2 → '{OUTPUT_PATH}'")


# Step 11: Merge datasets 1 and 2
logger.info("Merging datasets...")

if not DATASET1_PATH.exists():
    raise FileNotFoundError(f"Dataset 1 not found: {DATASET1_PATH}. Run dataset 1 script first.")

df1 = pd.read_csv(DATASET1_PATH)
df2 = pd.read_csv(OUTPUT_PATH)

# Align columns before concat to prevent silent misalignment
all_cols = list(dict.fromkeys(list(df1.columns) + list(df2.columns)))
df1      = df1.reindex(columns=all_cols)
df2      = df2.reindex(columns=all_cols)

df_combined = pd.concat([df1, df2], ignore_index=True)
df_combined = df_combined.drop_duplicates()
df_combined = df_combined.dropna(subset=["symptoms", "disease"])

logger.info(f"Combined dataset: {len(df_combined)} rows.")
print(df_combined.head())

df_combined.to_csv(COMBINED_PATH, index=False)
logger.info(f"Saved combined dataset → '{COMBINED_PATH}'")
print("\n── Sample combined output ──")
print(df_combined[["disease", "symptoms"]].head(10).to_string())