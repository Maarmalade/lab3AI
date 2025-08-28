import pandas as pd
import os

# ========================
# Load CSV files
# ========================
cases = pd.read_csv("cases_malaysia.csv", parse_dates=["date"])
deaths = pd.read_csv("deaths_malaysia.csv", parse_dates=["date"])
hospital = pd.read_csv("hospital.csv", parse_dates=["date"])
icu = pd.read_csv("icu.csv", parse_dates=["date"])
tests = pd.read_csv("tests_malaysia.csv", parse_dates=["date"])
vax = pd.read_csv("vax_malaysia.csv", parse_dates=["date"])

# ========================
# Select useful features
# ========================
cases = cases[["date", "cases_new", "cases_active"]]
deaths = deaths[["date", "deaths_new"]]
hospital = hospital.groupby("date", as_index=False)[["admitted_covid", "hosp_covid"]].sum()
icu = icu.groupby("date", as_index=False)[["icu_covid", "vent_covid"]].sum()
tests = tests.rename(columns={"rtk-ag": "rtk_ag"})[["date", "rtk_ag", "pcr"]]
vax = vax[[
    "date", "daily_partial", "daily_full", "daily_booster",
    "cumul_partial", "cumul_full", "cumul_booster"
]]

# ========================
# Merge datasets
# ========================
df = cases.merge(deaths, on="date", how="left")
df = df.merge(hospital, on="date", how="left")
df = df.merge(icu, on="date", how="left")
df = df.merge(tests, on="date", how="left")
df = df.merge(vax, on="date", how="left")

# ========================
# Add lag features (7 & 14 days)
# ========================
lag_features = [
    "cases_new", "cases_active", "admitted_covid", "hosp_covid",
    "icu_covid", "vent_covid", "rtk_ag", "pcr",
    "daily_partial", "daily_full", "daily_booster",
    "cumul_partial", "cumul_full", "cumul_booster"
]

for col in lag_features:
    df[f"{col}_lag7"] = df[col].shift(7)
    df[f"{col}_lag14"] = df[col].shift(14)

# Drop NaN rows
df = df.dropna().reset_index(drop=True)

# ========================
# Filter date range
# ========================
start_date = "2021-07-15"
end_date   = "2022-01-15"
df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].reset_index(drop=True)

# ========================
# Shuffle rows (ignore date order)
# ========================
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ========================
# Split into 6 parts
# ========================
n = len(df)
split_size = n // 6

splits = [df[i*split_size:(i+1)*split_size] for i in range(5)]
splits.append(df[5*split_size:])  # last split takes remainder

# ========================
# Save to folders
# ========================
os.makedirs("training_set", exist_ok=True)
os.makedirs("testing_set", exist_ok=True)

for i, split in enumerate(splits, 1):
    if i < 6:
        split.to_csv(f"training_set/set_{i}.csv", index=False)
    else:
        split.to_csv(f"testing_set/set_{i}.csv", index=False)

print("✅ Data successfully split and saved!")
print(f"Training sets: {split_size*5} rows, Testing set: {len(splits[-1])} rows")
