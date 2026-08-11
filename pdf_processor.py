from pathlib import Path
from pypdf import PdfReader


def extract_pdf_pages(pdf_path: str | Path) -> list[dict]:
    """
    Extract text page-by-page so every retrieved chunk can cite
    the original resume and page number.
    """
    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))

    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())

        if text.strip():
            pages.append({
                "text": text,
                "page": page_number,
                "file": pdf_path.name
            })

    return pages


def extract_candidate_name(pages: list[dict], filename: str) -> str:
    """
    Lightweight name extraction. For production use, replace this with
    a dedicated NER/parser if needed. We intentionally do not infer
    demographic attributes.
    """
    if pages:
        first_lines = pages[0]["text"].split()
        if len(first_lines) >= 2:
            candidate = " ".join(first_lines[:4])
            # Avoid treating common resume headings as a name.
            bad = {
                "resume", "curriculum vitae", "cv",
                "professional summary", "profile", "objective"
            }
            if candidate.lower() not in bad:
                return candidate

    return Path(filename).stem
