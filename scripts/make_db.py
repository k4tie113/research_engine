"""
make_db.py
----------
Takes papers.csv and saves it into papers.db (SQLite).
"""

import sqlite3, pandas as pd

df = pd.read_csv("metadata/papers.csv")

conn = sqlite3.connect("metadata/papers.db")
df.to_sql("papers", conn, if_exists="replace", index=False)
conn.close()

print("Database created at metadata/papers.db")
