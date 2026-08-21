import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier


# Load data
df = pd.read_csv(
    r"C:\Users\poppy\OneDrive\Documents\stellar-data-analysis"
    r"\raw\star_classification.csv"
)

# Clean missing values
df = df.replace(-9999, pd.NA)

# Colour indices
df["u_g"] = df["u"] - df["g"]
df["g_r"] = df["g"] - df["r"]
df["r_i"] = df["r"] - df["i"]
df["i_z"] = df["i"] - df["z"]


features = [
    "u",
    "g",
    "r",
    "i",
    "z",
    "u_g",
    "g_r",
    "r_i",
    "i_z"
]

df_model = df.dropna(subset=features + ["class"])

X = df_model[features]
y = df_model["class"]


# Random Forest
model = RandomForestClassifier(
    n_estimators=30,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)


# Five-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)


print("Cross-validation accuracy:")
print(scores.round(4))

print("\nMean accuracy:")
print(round(scores.mean(), 4))

print("\nStandard deviation:")
print(round(scores.std(), 4))