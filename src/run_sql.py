import sqlite3
import pandas as pd


connection = sqlite3.connect(
    r"C:\Users\poppy\OneDrive\Documents\stellar-data-analysis"
    r"\stellar_objects.db"
)

with open(
    r"C:\Users\poppy\OneDrive\Documents\stellar-data-analysis"
    r"\sql\01_basic_analysis.sql",
    "r"
) as file:
    sql = file.read()


queries = [
    query.strip()
    for query in sql.split(";")
    if query.strip()
]


for i, query in enumerate(queries, 1):

    print(f"\nQuery {i}")
    print("-" * 40)

    results = pd.read_sql_query(query, connection)

    print(results.to_string(index=False))


connection.close()

