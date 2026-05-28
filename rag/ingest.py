"""
rag/ingest.py — SEC EDGAR 10-K ingestion pipeline.

Downloads filings via sec-edgar-downloader, strips HTML/XBRL noise,
then hands raw text to the chunker.

Usage
-----
    from rag.ingest import fetch_10k

    # Returns list of {ticker, year, section, text} dicts
    chunks = fetch_10k("AAPL", years=[2022, 2023, 2024])

Dependencies (add to requirements.txt):
    sec-edgar-downloader>=5.0.2
    beautifulsoup4>=4.12
    lxml>=5.0
"""

from __future__ import annotations

import os
from pydoc import text
import re
import tempfile
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Lazy imports — heavy libs, only loaded when actually called
# ---------------------------------------------------------------------------


def _bs4():
    from bs4 import BeautifulSoup

    return BeautifulSoup


def _downloader():
    from sec_edgar_downloader import Downloader

    return Downloader


# ============================================================================
# PUBLIC API
# ============================================================================


def fetch_10k(
    ticker: str,
    years: list[int],
    data_dir: Optional[str] = None,
    identity: str = "FinSight research@finsight.local",
) -> list[dict]:
    """
    Download 10-K filings for `ticker` for each year in `years`,
    extract clean text, and return section chunks.

    Args:
        ticker:   Stock ticker, e.g. "AAPL"
        years:    List of fiscal years, e.g. [2022, 2023, 2024]
        data_dir: Directory to cache raw filings (default: ./data/sec_raw)
        identity: SEC EDGAR requires a name+email in the User-Agent header.

    Returns:
        List of chunk dicts:
        {
            "ticker":   "AAPL",
            "year":     2023,
            "form":     "10-K",
            "section":  "Item 1A",
            "title":    "Risk Factors",
            "text":     "...",
            "char_len": 4821,
        }
    """
    ticker = ticker.upper().strip()

    if data_dir is None:
        data_dir = str(Path(__file__).parent.parent / "data" / "sec_raw")
    os.makedirs(data_dir, exist_ok=True)

    from rag.chunker import chunk_filing

    all_chunks: list[dict] = []

    for year in years:
        print(f"  [ingest] Fetching {ticker} 10-K for {year}...")

        try:
            filing_text = _download_filing(ticker, year, data_dir, identity)
        except Exception as exc:
            print(f"  [ingest] ⚠ Could not fetch {ticker} {year}: {exc}")
            continue

        if not filing_text:
            print(f"  [ingest] ⚠ Empty filing for {ticker} {year}, skipping.")
            continue

        chunks = chunk_filing(
            text=filing_text,
            ticker=ticker,
            year=year,
            form="10-K",
        )
        print(f"  [ingest] ✓ {ticker} {year}")

        all_chunks.extend(chunks)

    print(f"  [ingest] Done. Total chunks: {len(all_chunks)}")
    return all_chunks


# ============================================================================
# DOWNLOAD HELPERS
# ============================================================================


def _download_filing(
    ticker: str,
    year: int,
    data_dir: str,
    identity: str,
) -> Optional[str]:
    """
    Download the 10-K for `ticker` filed in `year` using sec-edgar-downloader.
    Returns the cleaned plain-text of the filing document.
    """
    Downloader = _downloader()
    dl = Downloader(
        company_name="FinSight Research",
        email_address="ccalvin731@gmail.com",
        download_folder=data_dir,
    )
    print("DL: ", dl)

    # sec-edgar-downloader saves files under:
    #   {download_folder}/sec-edgar-filings/{ticker}/10-K/...
    filing_dir = Path(data_dir) / "sec-edgar-filings" / ticker / "10-K"

    # after_date / before_date filters to the fiscal year
    # 10-Ks for fiscal year N are typically filed in early N+1
    after = f"{year - 1}-01-01"
    before = f"{year + 1}-06-30"

    dl.get(
        "10-K",
        ticker,
        limit=1,
        after=after,
        before=before,
        download_details=True,
    )

    # Find the downloaded filing document (.htm or .txt)
    doc_path = _find_primary_document(filing_dir)
    if doc_path is None:
        raise FileNotFoundError(
            f"No 10-K document found in {filing_dir} for {ticker} {year}"
        )

    raw = doc_path.read_text(encoding="utf-8", errors="replace")
    return _strip_html(raw)


def _find_primary_document(filing_dir: Path) -> Optional[Path]:
    """
    Walk the filing directory and return the primary document.
    sec-edgar-downloader nests files under a CIK/accession-number subdirectory.
    The primary doc is usually the largest .htm file.
    """
    if not filing_dir.exists():
        return None

    candidates: list[tuple[int, Path]] = []

    for path in filing_dir.rglob("*"):
        if path.suffix.lower() in (".htm", ".html", ".txt") and path.is_file():
            # skip index files and exhibit files
            name = path.name.lower()
            if any(skip in name for skip in ("index", "exhibit", "ex-", "ex_")):
                continue
            candidates.append((path.stat().st_size, path))

    if not candidates:
        return None

    # Return the largest file — that's the main 10-K document
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def _strip_html(raw: str) -> str:
    """
    Remove HTML/XBRL tags and normalise whitespace.
    Falls back to regex stripping if BeautifulSoup is unavailable.
    """
    try:
        BeautifulSoup = _bs4()
        soup = BeautifulSoup(raw, "lxml")

        # Remove script, style, and XBRL inline tags
        for tag in soup(
            [
                "script",
                "style",
                "ix:nonfraction",
                "ix:nonnumeric",
                "ix:header",
                "xbrl",
                "link",
                "meta",
            ]
        ):
            tag.decompose()

        text = soup.get_text(separator="\n")
    except Exception:
        # Fallback: simple regex strip
        text = re.sub(r"<[^>]+>", " ", raw)

    # Normalise whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


if __name__ == "__main__":
    # Quick test
    chunks = fetch_10k("AAPL", years=[2022])
    print(f"Fetched {len(chunks)} chunks.")
