"""
pdf_to_text.py
--------------
Converts all PDFs in data/ into plain text files.
Each paper: 2501.12345.pdf → 2501.12345.txt
"""

from PyPDF2 import PdfReader
import os

for fname in os.listdir("data"):
    if fname.endswith(".pdf"):
        txt_path = fname.replace(".pdf", ".txt")
        out_path = f"data/{txt_path}"

        if os.path.exists(out_path):
            continue  # skip if already processed

        try:
            reader = PdfReader(f"data/{fname}")
            text = ""
            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"

            with open(out_path, "w") as f:
                f.write(text)
            print(f"📝 Converted {fname} → {txt_path}")
        except Exception as e:
            print(f"Error converting {fname}: {e}")

print("All PDFs converted to text")
