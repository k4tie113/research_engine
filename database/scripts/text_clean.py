import re

def basic_clean(text: str, drop_refs: bool = True) -> str:
    """
    Lightly normalize PDF text.
    - Removes multiple spaces
    - Strips line breaks
    - Optionally drops reference sections
    """
    if not text:
        return ""

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Optionally remove References section (very rough)
    if drop_refs:
        text = re.split(r'\bReferences\b', text, flags=re.IGNORECASE)[0]

    # Clean up weird artifacts
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")

    return text.strip()
