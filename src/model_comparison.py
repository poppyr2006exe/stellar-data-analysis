import pandas as pd

from sklearn.model_selection import train_test_split # Splits data into the 80% training / 20% testing sets
from sklearn.preprocessing import StandardScaler # prevents a feature simply having a larger numerical scale from disproportionately affecting the model
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#three algorithms comparing

from sklearn.metrics import accuracy_score, f1_score


df = pd.read_csv(
    r"C:\Users\poppy\OneDrive\Documents\stellar-data-analysis"
    r"\raw\star_classification.csv"
)

df = df.replace(-9999, pd.NA)


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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# testing on the same 20%

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC())
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


results = []

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1
    })

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")


results_df = pd.DataFrame(results)

print("\nModel comparison")
print("----------------")
print(results_df.to_string(index=False))




