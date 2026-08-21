import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


# Load data
df = pd.read_csv(
    r"C:\Users\poppy\OneDrive\Documents\stellar-data-analysis"
    r"\raw\star_classification.csv"
)

# Clean missing values
df = df.replace(-9999, pd.NA)

# Create colour indices
df["u_g"] = df["u"] - df["g"]
df["g_r"] = df["g"] - df["r"]
df["r_i"] = df["r"] - df["i"]
df["i_z"] = df["i"] - df["z"]


magnitudes = [
    "u",
    "g",
    "r",
    "i",
    "z"
]

colours = [
    "u_g",
    "g_r",
    "r_i",
    "i_z"
]

all_features = magnitudes + colours


# Train and test split
df_model = df.dropna(subset=all_features + ["class"])

y = df_model["class"]

X_train, X_test, y_train, y_test = train_test_split(
    df_model[all_features],
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Compare feature sets
feature_sets = {
    "Magnitudes only": magnitudes,
    "Colours only": colours,
    "Magnitudes + colours": all_features
}


for name, features in feature_sets.items():

    model = RandomForestClassifier(
        n_estimators=30,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train[features],
        y_train
    )

    predictions = model.predict(
        X_test[features]
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    print("\n" + name)
    print("-" * len(name))
    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(f1, 4))