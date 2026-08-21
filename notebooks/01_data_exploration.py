import pandas as pd
import matplotlib.pyplot as plt


# Load dataset
df = pd.read_csv(
    r"C:\Users\poppy\OneDrive\Documents\stellar-data-analysis"
    r"\raw\star_classification.csv"
)

print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

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


# The dataset uses -9999 as a missing value marker
print("\nNumber of -9999 values per column:")
print((df == -9999).sum())

df_clean = df.replace(-9999, pd.NA)

print("\nMissing values after cleaning:")
print(df_clean.isnull().sum())


# Create colour indices
df_clean["u_g"] = df_clean["u"] - df_clean["g"]
df_clean["g_r"] = df_clean["g"] - df_clean["r"]
df_clean["r_i"] = df_clean["r"] - df_clean["i"]
df_clean["i_z"] = df_clean["i"] - df_clean["z"]


print("\nMean colour indices by class:")
print(
    df_clean.groupby("class")[["u_g", "g_r", "r_i", "i_z"]]
    .mean()
    .round(3)
)

print("\nMedian colour indices by class:")
print(
    df_clean.groupby("class")[["u_g", "g_r", "r_i", "i_z"]]
    .median()
    .round(3)
)


print("\nu-band statistics by class:")
print(
    df_clean.groupby("class")["u"]
    .describe()
    .round(3)
)


# Class distribution
class_counts = df_clean["class"].value_counts()

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


# u-band distribution
plt.figure(figsize=(8, 5))

for object_class in ["STAR", "GALAXY", "QSO"]:
    values = df_clean[df_clean["class"] == object_class]["u"].dropna()

    plt.hist(
        values,
        bins=50,
        alpha=0.5,
        label=object_class
    )

plt.title("Distribution of u-band Magnitude by Object Class")
plt.xlabel("u-band magnitude")
plt.ylabel("Number of Observations")
plt.legend()
plt.tight_layout()

plt.savefig(
    "../figures/u_band_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Normalised g-r colour distribution
plt.figure(figsize=(8, 5))

for object_class in ["STAR", "GALAXY", "QSO"]:
    values = (
        df_clean[df_clean["class"] == object_class]["g_r"]
        .dropna()
        .astype(float)
    )

    plt.hist(
        values,
        bins=50,
        density=True,
        alpha=0.5,
        label=object_class
    )

plt.title("Normalised Distribution of g-r Colour by Object Class")
plt.xlabel("g-r colour index")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()

plt.savefig(
    "../figures/g_r_distribution_normalised.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Colour-colour diagram
plt.figure(figsize=(8, 6))

for object_class in ["STAR", "GALAXY", "QSO"]:
    subset = df_clean[
        df_clean["class"] == object_class
    ].dropna(subset=["g_r", "r_i"])

    plt.scatter(
        subset["g_r"].astype(float),
        subset["r_i"].astype(float),
        s=4,
        alpha=0.15,
        label=object_class
    )

plt.title("SDSS Colour-Colour Diagram")
plt.xlabel("g-r")
plt.ylabel("r-i")
plt.legend()
plt.tight_layout()

plt.savefig(
    "../figures/colour_colour_diagram.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Median numerical features
numerical_features = [
    "alpha", "delta", "u", "g", "r", "i", "z",
    "redshift", "u_g", "g_r", "r_i", "i_z"
]

print("\nMedian numerical features by class:")
print(
    df_clean.groupby("class")[numerical_features]
    .median()
    .round(3)
)


print("\nMedian redshift by class:")
print(
    df_clean.groupby("class")["redshift"]
    .median()
    .round(4)
)

print("\nRedshift statistics by class:")
print(
    df_clean.groupby("class")["redshift"]
    .describe()
    .round(4)
)


# Redshift boxplot
plt.figure(figsize=(8, 5))

df_clean.boxplot(
    column="redshift",
    by="class"
)

plt.title("Redshift Distribution by Object Class")
plt.suptitle("")
plt.xlabel("Object Class")
plt.ylabel("Redshift (z)")
plt.tight_layout()

plt.savefig(
    "../figures/redshift_by_class.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Zoomed redshift distribution
plt.figure(figsize=(8, 5))

for object_class in ["STAR", "GALAXY", "QSO"]:
    values = (
        df_clean[df_clean["class"] == object_class]["redshift"]
        .dropna()
        .astype(float)
    )

    plt.hist(
        values,
        bins=100,
        range=(-0.01, 2.0),
        density=True,
        alpha=0.5,
        label=object_class
    )

plt.title("Redshift Distribution by Object Class")
plt.xlabel("Redshift (z)")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()

plt.savefig(
    "../figures/redshift_distribution_zoomed.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()