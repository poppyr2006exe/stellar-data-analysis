import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay


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


# Features used by the model
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


# Training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Random Forest
model = RandomForestClassifier(
    n_estimators=30,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_test)


# Accuracy
print("Accuracy:")
print(round(accuracy_score(y_test, y_pred), 4))


# Classification report
print("\nClassification report:")
print(classification_report(y_test, y_pred))


# Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["GALAXY", "QSO", "STAR"]
)

plt.title("Random Forest Classification Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "../figures/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature importance:")
print(importance)


plt.figure(figsize=(8, 5))

importance.plot(kind="bar")

plt.title("Random Forest Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "../figures/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()