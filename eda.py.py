import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import ast
import logging
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Constants 
COMBINED_PATH  = Path("cleaned_dataset_combined.csv")
TOP_N_DISEASES = 20
TOP_N_SYMPTOMS = 20
TOP_DISEASES   = 12
TOP_SYMPTOMS   = 15
MIN_COOCCUR    = 2
PLOT_STYLE     = "seaborn-v0_8-whitegrid"

INDONESIAN_MARKERS = {
    "dengan", "dan", "atau", "pada", "yang", "untuk", "dari",
    "tidak", "ada", "ini", "itu", "juga", "dapat", "akan",
    "sebagai", "karena", "oleh", "dalam", "adalah", "serta",
    "bahwa", "bila", "jika", "seperti", "antara", "setelah"
}

plt.style.use(PLOT_STYLE)

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

def load_combined() -> pd.DataFrame:
    if not COMBINED_PATH.exists():
        raise FileNotFoundError("Combined dataset not found. Run preprocessing first.")
    df = pd.read_csv(COMBINED_PATH)
    df["symptoms_list"] = df["symptoms_list"].apply(parse_symptoms_list)
    df["symptom_count"] = df["symptoms_list"].apply(len)
    return df

def show(title: str):
    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()

# STEP 1 — Load & Overview
logger.info("── Step 1: Load & Overview ──")
df = load_combined()
logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

print("\n── Shape ──")
print(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")

print("\n── Column Types ──")
print(df.dtypes)

print("\n── Sample Rows ──")
print(df[["disease", "symptoms", "treatment", "risk"]].head(5).to_string())

print("\n── Unique Diseases ──")
print(f"Total unique diseases: {df['disease'].nunique()}")

# Missing values
print("\n── Missing Values ──")
missing     = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
print(pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct}))

fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="Reds", ax=ax)
ax.set_xlabel("Columns")
show("Missing Value Heatmap")

# Translation quality check (new — from MarianMT preprocessing)
print("\n── Translation Quality Check ──")
for col in ["symptoms", "treatment"]:
    if col not in df.columns:
        continue
    flagged     = df[col].apply(has_indonesian).sum()
    flagged_pct = round(flagged / len(df) * 100, 2)
    if flagged > 0:
        logger.warning(f"Column '{col}': {flagged} rows ({flagged_pct}%) may still contain Indonesian text.")
        print(df[df[col].apply(has_indonesian)][["disease", col]].head(5).to_string())
    else:
        logger.info(f"Column '{col}': ✓ No Indonesian markers detected.")

# STEP 2 — Disease Distribution
logger.info("── Step 2: Disease Distribution ──")

disease_counts = df["disease"].value_counts().head(TOP_N_DISEASES)

print("\n── Disease Distribution ──")
print(disease_counts.to_string())

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(
    disease_counts.index[::-1],
    disease_counts.values[::-1],
    color=sns.color_palette("viridis", len(disease_counts))
)
ax.set_xlabel("Number of Records")
ax.set_ylabel("Disease")
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

for bar, count in zip(bars, disease_counts.values[::-1]):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            str(count), va="center", fontsize=9)

show(f"Top {TOP_N_DISEASES} Disease Distribution")

# Class imbalance
print("\n── Class Imbalance Summary ──")
print(f"Most common  : '{disease_counts.index[0]}' ({disease_counts.iloc[0]} records)")
print(f"Least common : '{disease_counts.index[-1]}' ({disease_counts.iloc[-1]} records)")

imbalance_ratio = disease_counts.iloc[0] / disease_counts.iloc[-1]
print(f"Imbalance ratio : {imbalance_ratio:.2f}x")

if imbalance_ratio > 5:
    logger.warning("High class imbalance detected! Consider SMOTE or class weighting in model training.")
else:
    logger.info(f"Imbalance ratio {imbalance_ratio:.2f}x — within acceptable range.")

# STEP 3 — Symptom Frequency
logger.info("── Step 3: Symptom Frequency ──")

all_symptoms   = [s for sl in df["symptoms_list"] for s in sl]
symptom_counts = Counter(all_symptoms)
top_symptoms   = pd.Series(dict(symptom_counts.most_common(TOP_N_SYMPTOMS)))

print(f"\nTotal unique symptoms : {len(symptom_counts)}")
print(f"\nTop {TOP_N_SYMPTOMS} symptoms:")
print(top_symptoms.to_string())

# Bar chart
fig, ax = plt.subplots(figsize=(12, 6))
top_symptoms.sort_values().plot(
    kind="barh", ax=ax,
    color=sns.color_palette("viridis", TOP_N_SYMPTOMS)
)
ax.set_xlabel("Frequency")
ax.set_ylabel("Symptom")
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
show(f"Top {TOP_N_SYMPTOMS} Most Frequent Symptoms")

# Symptom count per record
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df["symptom_count"], bins=15, color="steelblue", edgecolor="white")
axes[0].set_xlabel("Number of Symptoms per Record")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Symptom Count Distribution")
axes[0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

axes[1].boxplot(
    df["symptom_count"], patch_artist=True,
    boxprops   =dict(facecolor="steelblue", color="navy"),
    medianprops=dict(color="red", linewidth=2)
)
axes[1].set_ylabel("Number of Symptoms")
axes[1].set_title("Symptom Count Boxplot")

show("Symptoms per Record")

print("\n── Symptom Count Stats ──")
print(df["symptom_count"].describe().round(2))

# New: flag suspiciously short symptom lists (likely bad translation)
short_records = df[df["symptom_count"] <= 1]
if len(short_records) > 0:
    logger.warning(
        f"{len(short_records)} records have only 1 symptom — "
        f"may indicate translation/parsing failure."
    )
    print("\n── Records with only 1 symptom ──")
    print(short_records[["disease", "symptoms"]].to_string())

# STEP 4 — Symptom–Disease Co-occurrence Heatmap
logger.info("── Step 4: Symptom–Disease Heatmap ──")

top_disease_names = df["disease"].value_counts().head(TOP_DISEASES).index.tolist()
top_symptom_names = [s for s, _ in symptom_counts.most_common(TOP_SYMPTOMS)]

df_top       = df[df["disease"].isin(top_disease_names)]
cooccurrence = pd.DataFrame(0, index=top_disease_names, columns=top_symptom_names)

for _, row in df_top.iterrows():
    for symptom in row["symptoms_list"]:
        if symptom in top_symptom_names:
            cooccurrence.loc[row["disease"], symptom] += 1

# Drop columns below minimum threshold
cooccurrence = cooccurrence.loc[:, (cooccurrence >= MIN_COOCCUR).any(axis=0)]

print("\n── Co-occurrence Matrix ──")
print(cooccurrence.to_string())

fig, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(
    cooccurrence, annot=True, fmt="d", cmap="YlOrRd",
    linewidths=0.5, linecolor="white",
    cbar_kws={"label": "Co-occurrence Count"}, ax=ax
)
ax.set_xlabel("Symptom")
ax.set_ylabel("Disease")
ax.tick_params(axis="x", rotation=45)
show("Symptom–Disease Co-occurrence Heatmap")

# STEP 5 — Data Quality Report
logger.info("── Step 5: Data Quality Report ──")

empty_symptoms  = (df["symptom_count"] == 0).sum()
empty_treatment = df["treatment"].apply(lambda x: str(x).strip() in ("", "nan")).sum()
empty_risk      = df["risk"].apply(lambda x: str(x).strip() in ("", "nan")).sum()
duplicate_rows  = df.duplicated(subset=["disease", "symptoms"]).sum()

# Translation quality per column
indonesian_symptoms  = df["symptoms"].apply(has_indonesian).sum()
indonesian_treatment = df["treatment"].apply(has_indonesian).sum() if "treatment" in df.columns else 0

print("\n════════════════════════════════════════")
print("         DATA QUALITY REPORT            ")
print("════════════════════════════════════════")
print(f"  Total records              : {len(df)}")
print(f"  Unique diseases            : {df['disease'].nunique()}")
print(f"  Unique symptoms            : {len(symptom_counts)}")
print(f"  Avg symptoms / record      : {df['symptom_count'].mean():.2f}")
print(f"  Records with no symptoms   : {empty_symptoms}")
print(f"  Records with no treatment  : {empty_treatment}")
print(f"  Records with no risk       : {empty_risk}")
print(f"  Duplicate rows             : {duplicate_rows}")
print(f"  ── Translation Quality ──")
print(f"  Symptoms w/ Indonesian text : {indonesian_symptoms}")
print(f"  Treatment w/ Indonesian text: {indonesian_treatment}")
print("════════════════════════════════════════\n")

if empty_symptoms > 0:
    logger.warning(f"{empty_symptoms} records have no symptoms — drop before model training.")
if empty_treatment > 0:
    logger.warning(f"{empty_treatment} records have no treatment — needs fallback logic.")
if empty_risk > 0:
    logger.warning(f"{empty_risk} records have no risk — severity scoring uses rule-based fallback.")
if duplicate_rows > 0:
    logger.warning(f"{duplicate_rows} duplicates found — re-run deduplication.")
if indonesian_symptoms > 0:
    logger.warning(f"{indonesian_symptoms} symptom rows still have Indonesian text — check MarianMT output.")
if indonesian_treatment > 0:
    logger.warning(f"{indonesian_treatment} treatment rows still have Indonesian text.")

logger.info("EDA complete!")