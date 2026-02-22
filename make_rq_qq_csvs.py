import pandas as pd
import re

FILE_PATH = "/Users/ishapatel/Machine Learning/ML_project/Machine-Learning-Project/data/bullets_2024.xlsx"

OUT_KQ = "KQ_model_ready.csv"
OUT_QQ = "QQ_model_ready.csv"

SHEETS = {
    "kq": "KQ responses (n=1575)",
    "qq": "QQ responses (n=1581)",
    "cset": "Cset (n=1240)",
}

# -------------------------
# Helpers
# -------------------------
def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()

def find_col(df, candidates):
    col_map = {norm(c): c for c in df.columns}
    for cand in candidates:
        if norm(cand) in col_map:
            return col_map[norm(cand)]
    return None

def keep_existing(df, cols):
    return df[[c for c in cols if c in df.columns]].copy()

def collapse_conclusion(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    if s in {"ID", "LeanID"}:
        return "Match"
    if s in {"Excl", "LeanExcl"}:
        return "NonMatch"
    if s in {"Insuff", "NoValue"}:
        return "Inconclusive"
    # fallback
    s2 = s.lower()
    if "incon" in s2 or "insuff" in s2 or "no value" in s2:
        return "Inconclusive"
    return pd.NA

# -------------------------
# Load sheets
# -------------------------
kq = pd.read_excel(FILE_PATH, sheet_name=SHEETS["kq"])
qq = pd.read_excel(FILE_PATH, sheet_name=SHEETS["qq"])
cset = pd.read_excel(FILE_PATH, sheet_name=SHEETS["cset"])

# -------------------------
# CSET: keep only the columns you requested
# -------------------------
cset_cset = find_col(cset, ["Cset"])
if cset_cset is None:
    raise ValueError("Couldn't find Cset column in Cset sheet.")

cset = cset.rename(columns={cset_cset: "Cset"})

cset_keep = [
    "Cset",
    "Type",
    "Mating",  # NOTE: ground truth — good for analysis, but remove for ML to avoid leakage
    "Assigns (Baseline)",
    "Assigns (Baseline+Repeat)",
    "Assigns (total)",
    "Q1 Quality (Voted)",
    "Q2 Quality (Voted)",
    "Cset Quality (voted)",
    "Q1 Quality (prescreen)",
    "Q2 Quality (prescreen)",
    "Cset Quality (prescreen)",
    "Subsession  Q2-K3",
    "Intervening bullets (min)",
    "Intervening bullets (max)",
    "Cset Category (Detailed)",
    "Caliber Q1",
    "Caliber Q2-K3 FAID",  # sometimes this is named slightly differently
    "Caliber Q2-K3",       # fallback if the FAID version isn't present
]
cset_small = keep_existing(cset, cset_keep)

# -------------------------
# Build KQ dataset
# -------------------------
kq_anon = find_col(kq, ["AnonID"])
kq_cset = find_col(kq, ["Cset", "KQset"])
kq_phase = find_col(kq, ["Phase"])
kq_concl = find_col(kq, ["K08_Conclusion", "Conclusion"])
kq_diff  = find_col(kq, ["K09_Difficulty", "Difficulty"])
kq_comp  = find_col(kq, ["K10_Comparability", "Comparability"])

if None in [kq_anon, kq_cset, kq_phase, kq_concl]:
    raise ValueError("Missing required columns in KQ responses. Print kq.columns to debug.")

kq_small = kq.rename(columns={
    kq_anon: "AnonID",
    kq_cset: "Cset",
    kq_phase: "Phase",
    kq_concl: "ConclusionRaw",
    (kq_diff or "KQ_Difficulty"): "Difficulty",
    (kq_comp or "KQ_Comparability"): "Comparability",
})

kq_small["ConclusionClass"] = kq_small["ConclusionRaw"].apply(collapse_conclusion)
kq_small = kq_small.dropna(subset=["ConclusionClass"])

kq_merged = kq_small.merge(cset_small, on="Cset", how="left")

# Drop AnonID from the final exported CSV (per your request)
kq_out = keep_existing(
    kq_merged,
    ["Cset", "Phase", "ConclusionRaw", "Difficulty", "Comparability", "ConclusionClass"] + cset_keep
).drop(columns=["AnonID"], errors="ignore")

kq_out.to_csv(OUT_KQ, index=False)
print(f"Saved {OUT_KQ} | rows={len(kq_out)} cols={kq_out.shape[1]}")

# -------------------------
# Build QQ dataset
# -------------------------
qq_anon = find_col(qq, ["AnonID"])
qq_cset = find_col(qq, ["Cset", "QQset"])
qq_phase = find_col(qq, ["Phase"])
qq_concl = find_col(qq, ["Q08_Conclusion", "Conclusion"])
qq_diff  = find_col(qq, ["Q09_Difficulty", "Difficulty"])
qq_comp  = find_col(qq, ["Q10_Comparability", "Comparability"])

if None in [qq_anon, qq_cset, qq_phase, qq_concl]:
    raise ValueError("Missing required columns in QQ responses. Print qq.columns to debug.")

qq_small = qq.rename(columns={
    qq_anon: "AnonID",
    qq_cset: "Cset",
    qq_phase: "Phase",
    qq_concl: "ConclusionRaw",
    (qq_diff or "QQ_Difficulty"): "Difficulty",
    (qq_comp or "QQ_Comparability"): "Comparability",
})

qq_small["ConclusionClass"] = qq_small["ConclusionRaw"].apply(collapse_conclusion)
qq_small = qq_small.dropna(subset=["ConclusionClass"])

qq_merged = qq_small.merge(cset_small, on="Cset", how="left")

qq_out = keep_existing(
    qq_merged,
    ["Cset", "Phase", "ConclusionRaw", "Difficulty", "Comparability", "ConclusionClass"] + cset_keep
).drop(columns=["AnonID"], errors="ignore")

qq_out.to_csv(OUT_QQ, index=False)
print(f"Saved {OUT_QQ} | rows={len(qq_out)} cols={qq_out.shape[1]}")