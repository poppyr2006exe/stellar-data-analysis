import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay


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


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Train model
model = RandomForestClassifier(
    n_estimators=30,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    labels=["GALAXY", "QSO", "STAR"]
)

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "../figures/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()