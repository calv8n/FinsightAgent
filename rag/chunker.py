"""
rag/chunker.py — Section-aware 10-K chunking.

Splits on Item headers (Item 1, 1A, 1B, 2 ... 15),
then sub-chunks large sections by paragraph to stay under embedding limits.
"""

from __future__ import annotations
import re

# All standard 10-K Item numbers in order
_ITEMS = [
    ("1", "Business"),
    ("1A", "Risk Factors"),
    ("1B", "Unresolved Staff Comments"),
    ("1C", "Cybersecurity"),
    ("2", "Properties"),
    ("3", "Legal Proceedings"),
    ("4", "Mine Safety Disclosures"),
    ("5", "Market for Registrant"),
    ("6", "Selected Financial Data"),
    ("7", "Management Discussion and Analysis"),
    ("7A", "Quantitative and Qualitative Disclosures"),
    ("8", "Financial Statements"),
    ("9", "Changes in and Disagreements"),
    ("9A", "Controls and Procedures"),
    ("9B", "Other Information"),
    ("10", "Directors and Executive Officers"),
    ("11", "Executive Compensation"),
    ("12", "Security Ownership"),
    ("13", "Certain Relationships"),
    ("14", "Principal Accountant Fees"),
    ("15", "Exhibits"),
]

# Regex that matches "Item 1A." / "ITEM 1A —" / "Item 1A:" etc.
_ITEM_RE = re.compile(
    r"(?:^|\n)\s*ITEM\s+(\d{1,2}[A-C]?)\s*[.\-:—]",
    re.IGNORECASE,
)

MAX_CHUNK_CHARS = 6_000  # ~1 500 tokens, safe for MiniLM-L6
MIN_CHUNK_CHARS = 120  # discard boilerplate stubs shorter than this


def chunk_filing(
    text: str,
    ticker: str,
    year: int,
    form: str = "10-K",
) -> list[dict]:
    """
    Split `text` into section chunks and return list of chunk dicts.

    Each dict:
        ticker, year, form, section, title, text, char_len
    """
    sections = _split_sections(text)
    chunks: list[dict] = []

    for item_num, section_text in sections.items():
        title = _item_title(item_num)
        label = f"Item {item_num}"

        for sub in _sub_chunk(section_text):
            if len(sub) < MIN_CHUNK_CHARS:
                continue
            chunks.append(
                {
                    "ticker": ticker,
                    "year": year,
                    "form": form,
                    "section": label,
                    "title": title,
                    "text": sub,
                    "char_len": len(sub),
                }
            )

    return chunks


# ── internals ────────────────────────────────────────────────────────────────


def _split_sections(text: str) -> dict[str, str]:
    """Return {item_number: section_text} preserving order."""
    matches = list(_ITEM_RE.finditer(text))
    if not matches:
        return {"0": text}

    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        item_num = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[item_num] = text[start:end].strip()

    return sections


def _sub_chunk(text: str) -> list[str]:
    """Break a section into ≤MAX_CHUNK_CHARS pieces, splitting on blank lines."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if current_len + len(para) > MAX_CHUNK_CHARS and current:
            chunks.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(para)
        current_len += len(para)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _item_title(item_num: str) -> str:
    for num, title in _ITEMS:
        if num == item_num.upper():
            return title
    return "Unknown"
