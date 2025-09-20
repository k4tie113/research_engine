#!/usr/bin/env python3
"""
run_all.py
----------
Master driver script to run the full local pipeline end-to-end:

    1. Query arXiv for the latest NLP (cs.CL) papers
       and write their metadata to data/metadata/papers.csv.

    2. Download each paper's PDF into data/pdfs/.

    3. Extract text from each PDF, break the text into
       overlapping ~800-token chunks, and write all chunks
       to data/chunks.jsonl.

This script simply executes the three existing scripts
(fetch_metadata.py, download_pdfs.py, chunk_pdfs.py)
in the correct order.
"""

import subprocess
from pathlib import Path

# --------------------------------------------------------------------
# Determine the directory that contains this file (scripts/).
# We will run the other scripts relative to this directory.
# --------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent

def run_script(script_name: str):
    """
    Helper to execute a Python script located in the same folder.

    Parameters
    ----------
    script_name : str
        File name of the script to run (for example 'fetch_metadata.py').

    Raises
    ------
    subprocess.CalledProcessError
        If the script returns a non-zero exit code.
    """
    print(f"\n=== Running {script_name} ===")
    # Use subprocess.run so that output is streamed directly to the console.
    subprocess.run(["python", str(SCRIPT_DIR / script_name)], check=True)

def main():
    """
    Run the three stages of the pipeline sequentially.
    If any stage fails, the script stops and prints the error.
    """
    run_script("fetch_metadata.py")   # Step 1: fetch metadata for 300 NLP papers
    run_script("download_pdfs.py")     # Step 2: download their PDFs
    run_script("chunk_pdfs.py")        # Step 3: create data/chunks.jsonl

    print("\nPipeline completed successfully.")
    print("• Metadata: data/metadata/papers.csv")
    print("• PDFs:     data/pdfs/")
    print("• Chunks:   data/chunks.jsonl")

if __name__ == "__main__":
    main()
