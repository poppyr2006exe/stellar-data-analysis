import pandas as pd

# Load the SDSS17 dataset
df = pd.read_csv(
    r"C:\Users\poppy\OneDrive\Documents\stellar-data-analysis\raw\star_classification.csv"
)

print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nUnique values per column:")
print(df.nunique())

print("\nClass distribution:")
print(df["class"].value_counts())

import matplotlib.pyplot as plt

# Class distribution
class_counts = df["class"].value_counts()

plt.figure(figsize=(8, 5))
class_counts.plot(kind="bar")
plt.title("Distribution of Astronomical Object Classes")
plt.xlabel("Object Class")
plt.ylabel("Number of Observations")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "../figures/class_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()