import re
import pandas as pd

FILE_PATH = "/Users/ishapatel/Machine Learning/ML_project/Machine-Learning-Project/data/bullets_2024.xlsx"
OUT_PATH = "bullet_pairs_model_ready.csv"

SHEETS = {
    "kq": "KQ responses (n=1575)",
    "qq": "QQ responses (n=1581)",
    "cset": "Cset (n=1240)",
    "participant": "Summary by Participant (n=49)",
}

# -------------------------
# Helpers
# -------------------------
def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()

def find_col(df, candidates):
    """
    Return the first column in df whose normalized name matches any candidate normalized name.
    candidates: list[str]
    """
    col_map = {norm(c): c for c in df.columns}
    for cand in candidates:
        if norm(cand) in col_map:
            return col_map[norm(cand)]
    return None

def collapse_conclusion(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()

    # Common values in these studies: ID, LeanID, Excl, LeanExcl, Insuff, NoValue
    if s in {"ID", "LeanID"}:
        return "Match"
    if s in {"Excl", "LeanExcl"}:
        return "NonMatch"
    if s in {"Insuff", "NoValue"}:
        return "Inconclusive"

    # If there are spelling variants, handle them here:
    s2 = s.lower()
    if "incon" in s2 or "insuff" in s2 or "no value" in s2:
        return "Inconclusive"
    return pd.NA

def keep_existing(df, cols):
    return df[[c for c in cols if c in df.columns]].copy()

# -------------------------
# Load sheets
# -------------------------
kq = pd.read_excel(FILE_PATH, sheet_name=SHEETS["kq"])
qq = pd.read_excel(FILE_PATH, sheet_name=SHEETS["qq"])
cset = pd.read_excel(FILE_PATH, sheet_name=SHEETS["cset"])
part = pd.read_excel(FILE_PATH, sheet_name=SHEETS["participant"], header=1)


# -------------------------
# Identify key columns in responses
# -------------------------
# These are common column patterns you showed: AnonID, Cset, Phase, *Conclusion, *Difficulty, *Comparability
kq_anon = find_col(kq, ["AnonID", "anonid"])
kq_cset = find_col(kq, ["Cset", "cset"])
kq_phase = find_col(kq, ["Phase", "phase"])
kq_concl = find_col(kq, ["K08_Conclusion", "Conclusion", "K08 Conclusion"])
kq_diff  = find_col(kq, ["K09_Difficulty", "Difficulty", "K09 Difficulty"])
kq_comp  = find_col(kq, ["K10_Comparability", "Comparability", "K10 Comparability"])

qq_anon = find_col(qq, ["AnonID", "anonid"])
qq_cset = find_col(qq, ["Cset", "cset", "QQset"])  # sometimes QQ uses QQset; we’ll still try
qq_phase = find_col(qq, ["Phase", "phase"])
qq_concl = find_col(qq, ["Q08_Conclusion", "Conclusion", "Q08 Conclusion"])
qq_diff  = find_col(qq, ["Q09_Difficulty", "Difficulty", "Q09 Difficulty"])
qq_comp  = find_col(qq, ["Q10_Comparability", "Comparability", "Q10 Comparability"])

# Sanity checks (fail fast with clear error)
for name, val in [
    ("kq_anon", kq_anon), ("kq_cset", kq_cset), ("kq_concl", kq_concl),
    ("qq_anon", qq_anon), ("qq_cset", qq_cset), ("qq_concl", qq_concl),
]:
    if val is None:
        raise ValueError(f"Could not find required column for {name}. "
                         f"Please print columns for that sheet and we’ll map it.")

# -------------------------
# Build a unified responses table (KQ + QQ)
# -------------------------
kq_small = kq.rename(columns={
    kq_anon: "AnonID",
    kq_cset: "Cset",
    (kq_phase or "Phase"): "Phase",
    kq_concl: "ConclusionRaw",
    (kq_diff or "KQ_Difficulty"): "Difficulty",
    (kq_comp or "KQ_Comparability"): "Comparability",
})
kq_small["ResponseType"] = "KQ"

qq_small = qq.rename(columns={
    qq_anon: "AnonID",
    qq_cset: "Cset",
    (qq_phase or "Phase"): "Phase",
    qq_concl: "ConclusionRaw",
    (qq_diff or "QQ_Difficulty"): "Difficulty",
    (qq_comp or "QQ_Comparability"): "Comparability",
})
qq_small["ResponseType"] = "QQ"

# Keep only columns that exist
kq_small = keep_existing(kq_small, ["AnonID", "Cset", "Phase", "ResponseType", "ConclusionRaw", "Difficulty", "Comparability"])
qq_small = keep_existing(qq_small, ["AnonID", "Cset", "Phase", "ResponseType", "ConclusionRaw", "Difficulty", "Comparability"])

responses = pd.concat([kq_small, qq_small], ignore_index=True)
responses["ConclusionClass"] = responses["ConclusionRaw"].apply(collapse_conclusion)

# Drop rows where we can’t map to your 3-class target
responses = responses.dropna(subset=["ConclusionClass"])

# -------------------------
# Clean participant-level table (examiner features)
# -------------------------
# We'll keep a reasonable starter set IF they exist.
# If you want more later, we can add.
part_anon = find_col(part, ["Participant (AnonID)", "AnonID", "Participant"])
if part_anon is None:
    print("Participant sheet columns:", list(part.columns))
    print(part.head(5))
    raise ValueError("Could not identify participant ID column in 'Summary by Participant'.")

part = part.rename(columns={part_anon: "AnonID"})

participant_keep = [
    "AnonID",
    "Baseline",
    "Overall",
    "Accuracy",
    "INC rate",
    "NV rate",
    "IncNV rate",
    "TID rate",
    "TEX rate",
    "FID rate",
    "FEX rate",
]
participant_small = keep_existing(part, participant_keep)

# IMPORTANT: If your goal is predicting classification, participant accuracy/rates can be “behavioral traits”
# but they can also be downstream of decisions. For interpretability, you may want to EXCLUDE them.
# We'll exclude them by default to avoid leakage.
participant_small = participant_small[["AnonID"]].copy()

# -------------------------
# Clean cset-level table (item features)
# -------------------------
cset_cset = find_col(cset, ["Cset", "cset"])
if cset_cset is None:
    raise ValueError("Could not find Cset column in 'Cset (n=1240)'. Please print cset.columns.")

cset = cset.rename(columns={cset_cset: "Cset"})

# Keep common useful features (only those that exist)
cset_keep = [
    "Cset",
    "Type",
    "Mating",
    "Cset Quality (voted)",
    "Cset Quality (prescreen)",
    "Intervening bullets (min)",
    "Intervening bullets (max)",
    "Caliber Q1",
    "Caliber Q2-K3 FAID",
    "Q1 Quality (voted)",
    "Q2 Quality (voted)",
    "Q1 Quality (prescreen)",
    "Q2 Quality (prescreen)",
    "Cset Category (Detailed)",
]
cset_small = keep_existing(cset, cset_keep)

# -------------------------
# Merge all together
# -------------------------
df = responses.merge(participant_small, on="AnonID", how="left")
df = df.merge(cset_small, on="Cset", how="left")

# -------------------------
# Save
# -------------------------
df.to_csv(OUT_PATH, index=False)
print(f"Saved: {OUT_PATH}")
print("Rows:", len(df), "Cols:", df.shape[1])
print(df["ConclusionClass"].value_counts())