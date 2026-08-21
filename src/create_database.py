import pandas as pd
import sqlite3


# Load the raw dataset
df = pd.read_csv(
    r"C:\Users\poppy\OneDrive\Documents\stellar-data-analysis"
    r"\raw\star_classification.csv"
)

# Replace the missing value marker
df = df.replace(-9999, pd.NA)

# Create the same colour indices used in the EDA
df["u_g"] = df["u"] - df["g"]
df["g_r"] = df["g"] - df["r"]
df["r_i"] = df["r"] - df["i"]
df["i_z"] = df["i"] - df["z"]

# Create SQLite database
connection = sqlite3.connect(
    r"C:\Users\poppy\OneDrive\Documents\stellar-data-analysis"
    r"\stellar_objects.db"
)

# Save the dataframe as a SQL table
df.to_sql(
    "observations",
    connection,
    if_exists="replace",
    index=False
)

# Check that the table was created correctly
cursor = connection.cursor()

cursor.execute(
    "SELECT COUNT(*) FROM observations"
)

number_of_rows = cursor.fetchone()[0]

print(f"Database created successfully.")
print(f"Number of observations: {number_of_rows}")

connection.close()

