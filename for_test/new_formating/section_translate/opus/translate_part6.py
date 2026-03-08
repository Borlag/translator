"""
Translate original_new_part6.docx (EN→RU) — Part 6 of CMM Main Landing Gear Leg.
Uses shared cmm_translation_lib for consistent terminology across all parts.
"""
from pathlib import Path
from cmm_translation_lib import translate_document

SRC = Path(r"C:\Users\Urdul\Desktop\project\translator\for_test\new_formating\section\original_new_part6.docx")
DST = Path(r"C:\Users\Urdul\Desktop\project\translator\for_test\new_formating\section_translate\opus\original_new_part6.docx")

if __name__ == "__main__":
    translate_document(src=str(SRC), dst=str(DST))
