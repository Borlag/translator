"""docxru - DOCX technical aviation EN->RU translation pipeline."""

from .pdf_pipeline import translate_pdf
from .pipeline import translate_docx

__all__ = ["translate_docx", "translate_pdf"]
