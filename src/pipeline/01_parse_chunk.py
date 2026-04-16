"""
Step 01 – Parse PDFs and chunk text into JSONL files.

Output: data/processed/<stem>.jsonl
Each line: {"chunk_id": str, "source": str, "page": int, "text": str}

OCR fallback: If a page yields < OCR_TEXT_THRESHOLD characters (scanned PDF),
the page is rendered as an image and passed to Tesseract (lang=vie+eng).
Requires: pytesseract + Pillow + Tesseract binary with Vietnamese language pack.
  - Windows : https://github.com/UB-Mannheim/tesseract/wiki
  - Linux   : sudo apt install tesseract-ocr tesseract-ocr-vie
  - macOS   : brew install tesseract tesseract-lang
"""

import argparse
import hashlib
import json
import re
import warnings
from pathlib import Path

import fitz  # PyMuPDF
import yaml

# ── Optional OCR deps ─────────────────────────────────────────────────────────
try:
    import pytesseract
    from PIL import Image
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False
    warnings.warn(
        "pytesseract / Pillow not installed — scanned PDFs will yield 0 chunks.\n"
        "Fix: pip install pytesseract Pillow  (+ install Tesseract binary)",
        stacklevel=1,
    )

# Pages with fewer characters than this after normal extraction trigger OCR
OCR_TEXT_THRESHOLD = 50
# Render resolution for OCR (higher = better quality, slower)
OCR_DPI = 200


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─── PDF → raw text per page ───────────────────────────────────────────────────

def _ocr_page(page: fitz.Page) -> str:
    """Render a PDF page to image and run Tesseract OCR (vie + eng)."""
    if not _OCR_AVAILABLE:
        return ""
    try:
        mat = fitz.Matrix(OCR_DPI / 72, OCR_DPI / 72)   # 72 DPI baseline
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img, lang="vie+eng", config="--psm 3")
        return text.strip()
    except Exception as exc:
        warnings.warn(f"OCR failed on page {page.number + 1}: {exc}")
        return ""


def extract_pages(pdf_path: Path) -> list[dict]:
    """Return list of {page, text} dicts for a PDF.

    For pages with very little embedded text (scanned/image PDFs),
    automatically falls back to Tesseract OCR (requires pytesseract + Pillow).
    """
    doc = fitz.open(str(pdf_path))
    pages = []
    ocr_pages: list[int] = []

    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        # ── OCR fallback for scanned pages ────────────────────────────────────
        if len(text) < OCR_TEXT_THRESHOLD:
            ocr_text = _ocr_page(page)
            if ocr_text:
                text = re.sub(r"\n{3,}", "\n\n", ocr_text).strip()
                ocr_pages.append(i)

        if text:
            pages.append({"page": i, "text": text})

    doc.close()

    if ocr_pages:
        print(f"    [OCR] {pdf_path.name}: applied OCR on pages {ocr_pages}")

    return pages


# ─── Legal structure splitter ─────────────────────────────────────────────────

# Matches beginning of Vietnamese legal section headers:
# "Điều 5.", "Điều 5:", "Chương I", "Chương II", "Mục 1", "Mục 2"
_LEGAL_HEADER = re.compile(
    r'(?m)(?=(?:Điều\s+\d+|Chương\s+[IVXLCDM0-9]+|Mục\s+[IVXLCDM0-9]+)[\s\.\:])'
)


def _split_legal_sections(text: str) -> list[str]:
    """Split text at Vietnamese legal section headers (Điều, Chương, Mục)."""
    parts = _LEGAL_HEADER.split(text)
    return [p.strip() for p in parts if p.strip()]


# ─── Sentence-aware chunker ───────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split Vietnamese text into sentences on punctuation and newline boundaries."""
    parts = re.split(r'(?<=[.!?\n])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def _sentence_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Sentence-level chunking fallback for oversized sections."""
    sentences = _split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len > chunk_size and current:
            chunks.append(" ".join(current).strip())
            # Build overlap from tail of current window
            tail: list[str] = []
            tail_len = 0
            for s in reversed(current):
                if tail_len + len(s) + 1 > overlap:
                    break
                tail.insert(0, s)
                tail_len += len(s) + 1
            current, current_len = tail, tail_len
        current.append(sent)
        current_len += sent_len + 1

    if current:
        chunks.append(" ".join(current).strip())
    return [c for c in chunks if c]


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into chunks respecting Vietnamese legal structure first,
    then falling back to sentence boundaries for oversized sections.
    """
    sections = _split_legal_sections(text)

    # If no legal headers found, fall back entirely to sentence chunking
    if len(sections) <= 1:
        return _sentence_chunk(text, chunk_size, overlap)

    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for section in sections:
        sec_len = len(section)

        if sec_len > chunk_size:
            # Section too big: flush accumulated current, then sentence-split this section
            if current_parts:
                chunks.append("\n\n".join(current_parts).strip())
                current_parts, current_len = [], 0
            sub_chunks = _sentence_chunk(section, chunk_size, overlap)
            chunks.extend(sub_chunks)
        elif current_len + sec_len + 2 > chunk_size and current_parts:
            # Would overflow: flush current and start new with overlap from last section
            chunks.append("\n\n".join(current_parts).strip())
            # Carry last section as overlap context if it fits
            last = current_parts[-1]
            if len(last) <= overlap:
                current_parts = [last, section]
                current_len = len(last) + sec_len + 2
            else:
                current_parts = [section]
                current_len = sec_len
        else:
            current_parts.append(section)
            current_len += sec_len + 2

    if current_parts:
        chunks.append("\n\n".join(current_parts).strip())

    return [c for c in chunks if c]


def make_chunk_id(source: str, page: int, idx: int) -> str:
    raw = f"{source}::{page}::{idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


# ─── Main ──────────────────────────────────────────────────────────────────────

def process_pdf(pdf_path: Path, out_dir: Path, chunk_size: int, overlap: int) -> int:
    pages = extract_pages(pdf_path)
    out_path = out_dir / (pdf_path.stem + ".jsonl")
    total = 0
    with out_path.open("w", encoding="utf-8") as f:
        for page_data in pages:
            chunks = chunk_text(page_data["text"], chunk_size, overlap)
            for idx, chunk in enumerate(chunks):
                record = {
                    "chunk_id": make_chunk_id(pdf_path.name, page_data["page"], idx),
                    "source": pdf_path.name,
                    "source_path": str(pdf_path),
                    "page": page_data["page"],
                    "chunk_index": idx,
                    "text": chunk,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1
    return total


def run(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    raw_dir = Path(cfg["data"]["raw_dir"])
    out_dir = Path(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = cfg["chunking"]["chunk_size"]
    overlap = cfg["chunking"]["chunk_overlap"]

    pdf_files = list(raw_dir.rglob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files.")

    total_chunks = 0
    skipped: list[str] = []

    for pdf in pdf_files:
        n = process_pdf(pdf, out_dir, chunk_size, overlap)
        status = f"{n} chunks" if n > 0 else "⚠️  0 chunks"
        print(f"  {pdf.name}: {status}")
        total_chunks += n
        if n == 0:
            skipped.append(pdf.name)

    print(f"\nDone. Total chunks: {total_chunks}")
    print(f"Output directory: {out_dir}")

    if skipped:
        print(f"\n⚠️  {len(skipped)} file(s) yielded 0 chunks (likely scanned PDFs with no OCR):")
        for name in skipped:
            print(f"   • {name}")
        if not _OCR_AVAILABLE:
            print("\n  → Install OCR support to fix this:")
            print("     pip install pytesseract Pillow")
            print("     # Then install Tesseract binary + Vietnamese language pack")
            print("     # Linux : sudo apt install tesseract-ocr tesseract-ocr-vie")
            print("     # macOS : brew install tesseract tesseract-lang")
            print("     # Windows: https://github.com/UB-Mannheim/tesseract/wiki")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse and chunk PDFs")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)
