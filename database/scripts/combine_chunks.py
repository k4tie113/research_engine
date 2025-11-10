"""
combine_chunks.py
Combines all chunk JSONL files (chunks1.jsonl, chunks2.jsonl, etc.)
from the chunks directory into one large file chunks_combined.jsonl.

Run this to merge all chunk files into a single file.
"""

from pathlib import Path
import json

# === CONFIGURATION ===
CHUNKS_DIR = Path(__file__).resolve().parents[1] / "data" / "chunks"
OUTPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "chunks_combined.jsonl"

def combine_chunks():
    # Find all chunk JSONL files
    chunk_files = sorted(CHUNKS_DIR.glob("chunks*.jsonl"))
    if not chunk_files:
        print("No chunk JSONL files found in chunks directory.")
        return

    print(f"🧩 Found {len(chunk_files)} chunk files.")
    print(f"📂 Combining into {OUTPUT_FILE.name} ...")

    total_entries = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        for chunk_file in chunk_files:
            print(f"  + Reading {chunk_file.name} ...", end=" ")
            file_entries = 0
            
            try:
                with open(chunk_file, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        line = line.strip()
                        if line:  # Skip empty lines
                            # Validate JSON before writing
                            try:
                                json.loads(line)
                                outfile.write(line + '\n')
                                file_entries += 1
                                total_entries += 1
                            except json.JSONDecodeError as e:
                                print(f"\n    ⚠️  Warning: Skipping invalid JSON line in {chunk_file.name}: {e}")
                
                print(f"{file_entries:,} entries")
            except Exception as e:
                print(f"\n    ❌ Error reading {chunk_file.name}: {e}")
                continue

    print(f"\n✅ Done! Combined {total_entries:,} entries into {OUTPUT_FILE}")

if __name__ == "__main__":
    combine_chunks()

