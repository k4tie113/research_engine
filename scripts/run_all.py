"""
run_all.py
----------
Runs the ingestion pipeline in order:
1. Fetch papers
2. Convert to text
3. Build DB
"""

import subprocess

def run_step(cmd, description):
    print(f"\n=== {description} ===")
    subprocess.run(["python"] + cmd, check=True)

if __name__ == "__main__":
    run_step(["scripts/fetch_papers.py"], "Step 1: Fetch papers")
    run_step(["scripts/pdf_to_text.py"], "Step 2: Convert PDFs to text")
    run_step(["scripts/make_db.py"], "Step 3: Build SQLite database")
    print("\n✅ All steps completed successfully!")
