#!/usr/bin/env python3
"""
Fix Russian translation issues in picture_ru2.pdf.

Issues fixed:
1. Double dots "ДИА.." → "ДИА." (199 occurrences)
2. Untranslated short English words (OF, IS, ON, TO, UP TO) in Russian text
3. Other context-specific translation fixes

The script modifies the CID-encoded overlay streams in the PDF.
"""

import re
import sys
import copy
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, DecodedStreamObject, EncodedStreamObject
import zlib


def build_cid_mapping(page):
    """Build CID→Unicode mapping from page's /arial font ToUnicode CMap."""
    resources = page['/Resources'].get_object()
    fonts = resources.get('/Font', {})
    if hasattr(fonts, 'get_object'):
        fonts = fonts.get_object()
    if '/arial' not in fonts:
        return None, None
    font = fonts['/arial'].get_object()
    if '/ToUnicode' not in font:
        return None, None
    cmap_data = font['/ToUnicode'].get_object().get_data().decode('latin-1', errors='replace')

    cid_to_unicode = {}
    in_bfrange = False
    in_bfchar = False
    for line in cmap_data.split('\n'):
        line = line.strip()
        if 'beginbfrange' in line:
            in_bfrange = True; continue
        if 'endbfrange' in line:
            in_bfrange = False; continue
        if 'beginbfchar' in line:
            in_bfchar = True; continue
        if 'endbfchar' in line:
            in_bfchar = False; continue
        if in_bfrange:
            m = re.match(r'<([0-9a-fA-F]+)>\s*<([0-9a-fA-F]+)>\s*<([0-9a-fA-F]+)>', line)
            if m:
                s, e, u = int(m.group(1), 16), int(m.group(2), 16), int(m.group(3), 16)
                for i in range(e - s + 1):
                    cid_to_unicode[s + i] = chr(u + i)
        if in_bfchar:
            m = re.match(r'<([0-9a-fA-F]+)>\s*<([0-9a-fA-F]+)>', line)
            if m:
                cid_to_unicode[int(m.group(1), 16)] = chr(int(m.group(2), 16))

    # Build reverse mapping (Unicode → CID)
    unicode_to_cid = {}
    for cid, uc in cid_to_unicode.items():
        if uc not in unicode_to_cid:
            unicode_to_cid[uc] = cid

    return cid_to_unicode, unicode_to_cid


def decode_hex_text(hex_str, cid_to_unicode):
    """Decode CID hex string to Unicode text."""
    decoded = ''
    for i in range(0, len(hex_str), 4):
        if i + 4 <= len(hex_str):
            cid = int(hex_str[i:i+4], 16)
            decoded += cid_to_unicode.get(cid, f'\ufffd')
    return decoded


def encode_text_to_hex(text, unicode_to_cid):
    """Encode Unicode text back to CID hex string."""
    hex_parts = []
    for ch in text:
        cid = unicode_to_cid.get(ch)
        if cid is None:
            # Try to find a close match or skip
            print(f"  WARNING: No CID for U+{ord(ch):04X} ({ch!r}), skipping", file=sys.stderr)
            continue
        hex_parts.append(f'{cid:04x}')
    return ''.join(hex_parts)


def normalize_spaces(text):
    """Replace non-breaking spaces (U+00A0) with regular spaces for matching."""
    return text.replace('\xa0', ' ')


def restore_nbsp(text):
    """Replace regular spaces back to non-breaking spaces (used in PDF overlays)."""
    return text.replace(' ', '\xa0')


def fix_text(text):
    """Apply all text fixes to a decoded Russian overlay string.
    Returns (fixed_text, was_changed).
    """
    original = text

    # Normalize non-breaking spaces to regular spaces for matching
    # The PDF uses U+00A0 (non-breaking space) instead of U+0020
    text = normalize_spaces(text)

    # === Fix 1: Double dots ДИА.. → ДИА. ===
    text = re.sub(r'ДИА\.\.(?!\.)', 'ДИА.', text)

    # === Fix 2: Untranslated "TO" in various contexts ===
    # "КРАСКА TO" → "ОКРАСКА ПО"
    text = text.replace('КРАСКА TO', 'ОКРАСКА ПО')
    # "И КРАСКА TO" → "И ОКРАСКА ПО"
    text = text.replace('И КРАСКА TO', 'И ОКРАСКА ПО')

    # "UP TO ВТУЛКА" → "ДО ВТУЛКИ"
    text = text.replace('UP TO ВТУЛКА', 'ДО ВТУЛКИ')
    # "UP TO БУРТИКОВ" etc
    text = re.sub(r'UP TO\b', 'ДО', text)

    # "TO A ГЛУБИНА" → "НА ГЛУБИНУ"
    text = text.replace('TO A ГЛУБИНА', 'НА ГЛУБИНУ')

    # "ЗАВЕРШЕНИЕ TO M-DLPS..." → "ЗАВЕРШЕНИЕ ПО M-DLPS..."
    text = re.sub(r'ЗАВЕРШЕНИЕ TO\b', 'ЗАВЕРШЕНИЕ ПО', text)

    # "ХРОМИРОВАНИЕ TO ОКАНЧИВАТЬСЯ" → "ХРОМИРОВАНИЕ ДО ЗАВЕРШЕНИЯ"
    text = text.replace('ХРОМИРОВАНИЕ TO ОКАНЧИВАТЬСЯ', 'ХРОМИРОВАНИЕ ДО ЗАВЕРШЕНИЯ')

    # "ХРОМИРОВАНИЕ TO 20,120mm" → "ХРОМИРОВАНИЕ ДО 20,120mm"
    text = re.sub(r'ХРОМИРОВАНИЕ TO\b', 'ХРОМИРОВАНИЕ ДО', text)

    # "TO ТОЧКА 'C'" → "ДО ТОЧКИ 'C'"
    text = text.replace('TO ТОЧКА', 'ДО ТОЧКИ')

    # "TO ПОДРЕЗКА" → "ДО ПОДРЕЗКИ"
    text = text.replace('TO ПОДРЕЗКА', 'ДО ПОДРЕЗКИ')

    # "TO M-DLPS..." → "ПО M-DLPS..."  (standalone TO before spec references)
    text = re.sub(r'\bTO (M[\-\xad]DLP)', r'ПО \1', text)

    # "БЕЗ КАДМИРОВАНИЯ TO" → "БЕЗ КАДМИРОВАНИЯ"
    text = text.replace('БЕЗ КАДМИРОВАНИЯ TO', 'БЕЗ КАДМИРОВАНИЯ')

    # === Fix 3: Untranslated "OF" ===
    text = text.replace('OF КОРПУС СТОЙКИ', 'КОРПУСА СТОЙКИ')
    text = text.replace('ДЛИНА OF', 'ДЛИНА')
    text = text.replace('OF ХРОМИРОВАНИЕ', 'ХРОМИРОВАНИЯ')
    text = text.replace('ПОВЕРХНОСТЬ OF ФЛАНЕЦ', 'ПОВЕРХНОСТЬ ФЛАНЦА')
    text = text.replace('ТИПИЧНО УСТАНОВКА OF', 'ТИПИЧНАЯ УСТАНОВКА')
    text = text.replace('OF ВТУЛКИ', 'ВТУЛКИ')

    # === Fix 4: Untranslated "IS" ===
    text = text.replace('ПОКРЫТИЕ IS НАНЕСЕНО', 'ПОКРЫТИЕ НАНЕСЕНО')
    text = text.replace('IS НАНЕСЕНО', 'НАНЕСЕНО')

    # === Fix 5: Untranslated "ON" ===
    text = text.replace('ПЕРЕКРЫТИЕ ON ХРОМИРОВАНИЕ РАДИУС', 'ПЕРЕКРЫТИЕ НА ХРОМИРОВАННОМ РАДИУСЕ')
    text = text.replace('КРАСКА ON ЭТА', 'ОКРАСКА НА ЭТОЙ')
    text = text.replace('ON ФАСКА', 'НА ФАСКЕ')

    # Restore non-breaking spaces
    text = restore_nbsp(text)

    return text, text != original


def fix_stream_data(stream_data, cid_to_unicode, unicode_to_cid):
    """Fix text in a single overlay content stream.
    Returns (new_data_bytes, was_changed).
    """
    text = stream_data.decode('latin-1', errors='replace')

    # Check if this stream has any hex text to fix
    hex_pattern = re.compile(r'<([0-9a-fA-F]+)>')
    matches = list(hex_pattern.finditer(text))

    if not matches:
        return stream_data, False

    changed = False
    new_text = text

    for match in reversed(matches):  # reverse to preserve positions
        hex_str = match.group(1)
        decoded = decode_hex_text(hex_str, cid_to_unicode)
        fixed, was_fixed = fix_text(decoded)

        if was_fixed:
            new_hex = encode_text_to_hex(fixed, unicode_to_cid)
            # Replace in the stream
            start, end = match.start(1), match.end(1)
            new_text = new_text[:start] + new_hex + new_text[end:]
            changed = True

    if changed:
        return new_text.encode('latin-1'), True
    return stream_data, False


def main():
    input_path = r'C:\Users\Urdul\Desktop\project\translator\for_test\new_formating\picture_ru.pdf'
    output_path = r'C:\Users\Urdul\Desktop\project\translator\for_test\new_formating\picture_ru_fixed_v2.pdf'

    print(f"Reading {input_path}...")
    reader = PdfReader(input_path)
    writer = PdfWriter()

    # Clone all pages
    writer.clone_reader_document_root(reader)

    total_fixes = 0
    pages_fixed = 0

    for page_idx in range(len(writer.pages)):
        page = writer.pages[page_idx]

        # Build mapping from this page's font
        cid_to_unicode, unicode_to_cid = build_cid_mapping(page)
        if not cid_to_unicode:
            continue

        contents = page.get('/Contents')
        if not contents:
            continue

        obj = contents.get_object() if hasattr(contents, 'get_object') else contents

        try:
            items = list(obj)
        except TypeError:
            continue

        page_changed = False

        # Process overlay streams (skip stream 0 which is original content)
        for stream_idx in range(1, len(items)):
            stream_ref = items[stream_idx]
            stream_obj = stream_ref.get_object()

            try:
                raw_data = stream_obj.get_data()
            except Exception:
                continue

            new_data, was_changed = fix_stream_data(raw_data, cid_to_unicode, unicode_to_cid)

            if was_changed:
                total_fixes += 1
                page_changed = True

                # Replace the stream data
                stream_obj.set_data(new_data)

        if page_changed:
            pages_fixed += 1

    print(f"\nFixed {total_fixes} streams across {pages_fixed} pages")
    print(f"Saving to {output_path}...")

    with open(output_path, 'wb') as f:
        writer.write(f)

    print("Done!")


if __name__ == '__main__':
    main()
