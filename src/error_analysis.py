import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv(
    r"C:\Users\poppy\OneDrive\Documents\stellar-data-analysis"
    r"\raw\star_classification.csv"
)

# Replace missing-value markers
df = df.replace(-9999, pd.NA)


# Create colour indices
df["u_g"] = df["u"] - df["g"]
df["g_r"] = df["g"] - df["r"]
df["r_i"] = df["r"] - df["i"]
df["i_z"] = df["i"] - df["z"]

# Features used by the Random Forest
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


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# Add the predictions to the test set
test_results = X_test.copy()
test_results["actual"] = y_test
test_results["predicted"] = y_pred

# Find stars that were incorrectly classified as galaxies
star_as_galaxy = test_results[
    (test_results["actual"] == "STAR") &
    (test_results["predicted"] == "GALAXY")
]

print("\nStars incorrectly classified as galaxies:")
print(f"Number: {len(star_as_galaxy)}")

# Correctly classified stars
correct_stars = test_results[
    (test_results["actual"] == "STAR") &
    (test_results["predicted"] == "STAR")
]

print("\nCorrectly classified stars:")
print(f"Number: {len(correct_stars)}")

# Colour indices to compare
colour_features = [
    "u_g",
    "g_r",
    "r_i",
    "i_z"
]

print("\nMean colour indices:")
print("--------------------")

print("\nCorrectly classified stars:")
print(correct_stars[colour_features].mean())

print("\nStars classified as galaxies:")
print(star_as_galaxy[colour_features].mean())

plt.figure(figsize=(8, 5))

plt.hist(
    correct_stars["r_i"],
    bins=40,
    alpha=0.6,
    label="Correctly classified STAR"
)

plt.hist(
    star_as_galaxy["r_i"],
    bins=40,
    alpha=0.6,
    label="STAR classified as GALAXY"
)

plt.xlabel("r-i colour index")
plt.ylabel("Number of objects")
plt.title("r-i Distribution for Correctly Classified and Misclassified Stars")
plt.legend()
plt.tight_layout()

plt.savefig(
    "../figures/star_galaxy_error_r_i.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()