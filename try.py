import pandas as pd
import os
import math

# =========================
# CONFIG
# =========================
BASE_PATH = r"F:\lab3AI"
OUTPUT_TRAIN = os.path.join(BASE_PATH, "training_set")
OUTPUT_TEST = os.path.join(BASE_PATH, "testing_set")

START_DATE = "2021-07-15"
END_DATE   = "2022-01-14"

os.makedirs(OUTPUT_TRAIN, exist_ok=True)
os.makedirs(OUTPUT_TEST, exist_ok=True)

# =========================
# LOAD & FILTER COLUMNS
# =========================
cases = pd.read_csv(os.path.join(BASE_PATH, "cases_malaysia.csv"), parse_dates=["date"])[["date", "cases_new"]]
deaths = pd.read_csv(os.path.join(BASE_PATH, "deaths_malaysia.csv"), parse_dates=["date"])[["date", "deaths_new"]]

# Merge only necessary columns
df = cases.merge(deaths, on="date", how="left")

# =========================
# FILTER DATE RANGE
# =========================
df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].reset_index(drop=True)

# =========================
# FEATURE ENGINEERING
# =========================
def create_lag_features(df, col, lags=[7,14]):
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df

important_cols = ["cases_new"]
for col in important_cols:
    df = create_lag_features(df, col)

df["cases_new_7d_avg"] = df["cases_new"].rolling(7).mean()

# Drop rows with NaN
df = df.dropna().reset_index(drop=True)

# =========================
# SHUFFLE & SPLIT INTO 6 SETS
# =========================
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

n_sets = 6
chunk_size = math.ceil(len(df_shuffled) / n_sets)

for set_number in range(1, n_sets+1):
    start = (set_number - 1) * chunk_size
    end = start + chunk_size
    chunk = df_shuffled.iloc[start:end]

    if set_number < n_sets:  # set 1–5 → training
        chunk.to_csv(os.path.join(OUTPUT_TRAIN, f"set{set_number}.csv"), index=False)
    else:  # set 6 → testing
        chunk.to_csv(os.path.join(OUTPUT_TEST, f"set{set_number}.csv"), index=False)

print("✅ Processing complete. 6 sets created (set1–5 in training, set6 in testing).")
