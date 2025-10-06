# scripts/text_clean.py
from __future__ import annotations
import regex as re
from ftfy import fix_text
from unidecode import unidecode

REFS_PAT = re.compile(r'(?mi)^(references|bibliography)\s*$', re.MULTILINE)
PAGE_NUM_PAT = re.compile(r'^\s*\d+\s*$', re.MULTILINE)
MULTISPACE = re.compile(r'[ \t]+')
HARD_HYPHEN = re.compile(r'(\p{Letter})-\n(\p{Letter})')  # word-↵wrap
LINEBREAK_IN_SENT = re.compile(r'(?<!\.)\n(?!\n)')  # single \n not paragraph
LIGATURES = str.maketrans({'\u00AD': '', '\u00A0': ' '})  # soft hyphen, nbsp

def strip_headers_footers(text: str) -> str:
    # kill obvious page-only lines (page numbers/headers)
    t = PAGE_NUM_PAT.sub('', text)
    return t

def remove_references_section(text: str) -> str:
    # if a References/Bibliography heading exists near the end, truncate there
    # find last occurrence
    matches = list(REFS_PAT.finditer(text))
    if matches:
        cut = matches[-1].start()
        # only cut if heading appears in last 40% of doc (avoid false positives)
        if cut > len(text) * 0.6:
            return text[:cut]
    return text

def normalize_whitespace(text: str) -> str:
    t = text.replace('\r\n', '\n').replace('\r', '\n')
    # un-break hyphenated words at EOL
    t = HARD_HYPHEN.sub(r'\1\2', t)
    # collapse stray linebreaks inside sentences
    t = LINEBREAK_IN_SENT.sub(' ', t)
    # collapse spaces
    t = MULTISPACE.sub(' ', t)
    # collapse >2 newlines to exactly 2 (paragraph break)
    t = re.sub(r'\n{3,}', '\n\n', t)
    # trim
    return t.strip()

def basic_clean(text: str, drop_refs: bool = True) -> str:
    if not text:
        return ""
    t = fix_text(text)  # fix mojibake
    t = t.translate(LIGATURES)
    t = strip_headers_footers(t)
    if drop_refs:
        t = remove_references_section(t)
    t = normalize_whitespace(t)
    # normalize to ascii-ish without killing math completely
    # (comment out if you want to preserve accents)
    t = unidecode(t)
    return t
