"""
combine_csv.py
Combines all yearly OAI-PMH metadata CSVs (papers_YYYY.csv)
into one large file papers_oai_combined.csv.

Run this AFTER all yearly fetch scripts finish.
"""

import pandas as pd
from pathlib import Path

# === CONFIGURATION ===
METADATA_DIR = Path(__file__).resolve().parents[1] / "data" / "metadata"
OUTPUT_FILE = METADATA_DIR / "papers_oai_combined.csv"

def combine_csvs():
    # Find all yearly CSVs
    csv_files = sorted(METADATA_DIR.glob("papers_*.csv"))
    if not csv_files:
        print("No yearly CSVs found. Make sure you’ve run the fetchers first.")
        return

    print(f"🧩 Found {len(csv_files)} yearly CSV files.")
    print(f"📂 Combining into {OUTPUT_FILE.name} ...")

    combined = []
    for file in csv_files:
        print(f"  + Reading {file.name}")
        df = pd.read_csv(file)
        combined.append(df)

    final_df = pd.concat(combined, ignore_index=True)

    # Remove potential duplicates (same arXiv ID appearing in multiple sets)
    if "id" in final_df.columns:
        final_df.drop_duplicates(subset=["id"], inplace=True)

    # Save combined CSV
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Done! Saved {len(final_df):,} unique papers to {OUTPUT_FILE}")

if __name__ == "__main__":
    combine_csvs()
