#!/usr/bin/env python3
"""
Fix remaining visual issues in picture_ru_fixed_v2.pdf:
1. Scale down wide Russian CID text overlays using Tz (horizontal scaling)
2. Add missing DIA/DIAMETER translations where overlays don't exist
3. Fix grammar issues (ДО ТОЧКА -> ДО ТОЧКИ)

Approach: modify EXISTING CID text streams to add horizontal compression,
rather than adding new white rects + text on top.
"""

import fitz
import os
import sys
import re
from pypdf import PdfReader, PdfWriter


INPUT_PATH = r'C:\Users\Urdul\Desktop\project\translator\for_test\new_formating\picture_ru_fixed_v2.pdf'
OUTPUT_PATH = r'C:\Users\Urdul\Desktop\project\translator\for_test\new_formating\picture_ru_fixed_v5.pdf'
FONT_PATH = r'C:\Windows\Fonts\arial.ttf'

DIA_KEYWORDS = ['DIA.', 'DIA', 'DIAMETER']
SKIP_PATTERNS = ['SAFRAN', 'CAGE:', 'PART No.', 'COMPONENT MAINTENANCE MANUAL',
                 'MAIN LANDING GEAR LEG', 'Figure', 'Page ', 'DIAPHRAGM']


def rects_overlap(b1, b2, tol=5):
    x_overlap = (b1[0] - tol) < b2[2] and (b2[0] - tol) < b1[2]
    y_overlap = (b1[1] - tol) < b2[3] and (b2[1] - tol) < b1[3]
    return x_overlap and y_overlap


def is_dia_text(text):
    text_upper = text.upper().strip()
    if not text_upper:
        return False
    for pat in SKIP_PATTERNS:
        if pat in text:
            return False
    if len(text_upper) <= 2 and text_upper.isalpha():
        return False
    if all(c in '0123456789.,- ()' for c in text_upper):
        return False
    for kw in DIA_KEYWORDS:
        if kw in text_upper:
            if 'DIAPHRAGM' in text_upper:
                continue
            return True
    return False


def translate_dia_text(text):
    t = text.strip()
    translations = [
        ('DIAMETER AFTER GRINDING THE CHROMIUM PLATE', 'ДИАМ. ПОСЛЕ ШЛИФОВАНИЯ ХРОМ. ПОКРЫТИЯ'),
        ('DIAMETER BEFORE CHROMIUM PLATE', 'ДИАМЕТР ДО ХРОМИРОВАНИЯ'),
        ('DIAMETER AFTER GRINDING', 'ДИАМЕТР ПОСЛЕ ШЛИФОВАНИЯ'),
        ('DIAMETER AFTER MACHINING', 'ДИАМЕТР ПОСЛЕ ОБРАБОТКИ'),
        ('DIAMETER BEFORE NICKEL', 'ДИАМЕТР ДО НИКЕЛИРОВАНИЯ'),
        ('DIAMETER BEFORE SULPHAMATE', 'ДИАМ. ДО СУЛЬФАМАТ. ПОКР.'),
        ('DIAMETER OVER LENGTH', 'ДИАМЕТР ПО ДЛИНЕ'),
        ('DIAMETER THRU BORE', 'ДИАМЕТР СКВОЗНОГО ОТВЕРСТИЯ'),
        ('DIAMETER MINIMUM', 'МИНИМАЛЬНЫЙ ДИАМЕТР'),
        ('DIAMETER REF.', 'ДИАМЕТР СПРАВ.'),
        ('DIAMETER BEFORE', 'ДИАМЕТР ДО'),
        ('DIAMETER AFTER', 'ДИАМЕТР ПОСЛЕ'),
        ('DIAMETERS THRU BORES', 'ДИАМЕТРЫ СКВОЗНЫХ ОТВЕРСТИЙ'),
        ('(INNER DIAMETER)', '(ВНУТР. ДИАМЕТР)'),
        ('(2 DIAMETERS)', '(2 ДИАМЕТРА)'),
        ('OUTSIDE DIAMETER', 'НАРУЖНЫЙ ДИАМЕТР'),
        ('MINIMUM DIAMETER', 'МИНИМАЛЬНЫЙ ДИАМЕТР'),
        ('PLATED DIAMETER', 'ПОКРЫТЫЙ ДИАМЕТР'),
        ('(DIAMETER)', '(ДИАМЕТР)'),
        ('DIAMETERS', 'ДИАМЕТРЫ'),
        ('DIAMETER', 'ДИАМЕТР'),
        ('BARREL OUTER DIA. LOWER', 'НАРУЖН. ДИАМ. ЦИЛИНДРА НИЖНИЙ'),
        ('BARREL OUTER DIA. UPPER', 'НАРУЖН. ДИАМ. ЦИЛИНДРА ВЕРХНИЙ'),
        ('DIA. BEFORE SULPHAMATE', 'ДИАМ. ДО СУЛЬФАМАТ. ПОКР.'),
        ('DIA. AFTER GRINDING CHROME', 'ДИАМ. ПОСЛЕ ШЛИФОВАНИЯ ХРОМА'),
        ('DIA. AFTER CHROME PLATING', 'ДИАМ. ПОСЛЕ ХРОМИРОВАНИЯ'),
        ('DIA. AFTER GRINDING', 'ДИАМ. ПОСЛЕ ШЛИФОВАНИЯ'),
        ('DIA. AFTER', 'ДИАМ. ПОСЛЕ'),
        ('DIA. MAX.', 'ДИАМ. МАКС.'),
        ('DIA. REF.', 'ДИАМ. СПРАВ.'),
        ('SPOTFACE DIA.', 'ДИАМ. ЦЕКОВКИ'),
        ('WORKING DIA.', 'РАБОЧИЙ ДИАМ.'),
        ('THROUGH DIA.', 'СКВОЗНОЙ ДИАМ.'),
        ('BORE DIA.', 'ДИАМ. ОТВЕР.'),
        ('AND DIA', 'И ДИАМ.'),
        ('DIA. SPOTFACE.', 'ДИАМ. ЦЕКОВКИ.'),
        ('DIA.', 'ДИАМ.'),
        ('DIA', 'ДИАМ.'),
    ]
    result = t
    for eng, rus in translations:
        if eng in result:
            result = result.replace(eng, rus)
    return result


# ======================================================================
# Phase 1: Scale down wide Russian CID text using Tz in existing streams
# ======================================================================

def phase1_scale_cid_streams(input_path, output_path):
    """
    For each page:
    1. Use fitz to get Russian vs English bbox widths
    2. Build a map of (approx_pdf_y -> scale_factor) for wide overlays
    3. Use pypdf to modify CID text streams, inserting Tz scaling
    """

    # Step 1: Gather scale factors from fitz
    doc = fitz.open(input_path)
    page_scales = {}  # page_idx -> list of (pdf_y_approx, scale_pct, ru_text)

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        cropbox = page.cropbox
        cropbox_x0 = float(cropbox.x0)
        cropbox_y1 = float(cropbox.y1)

        blocks = page.get_text('dict')['blocks']
        all_spans = []
        for b in blocks:
            if 'lines' not in b:
                continue
            for line in b['lines']:
                for span in line['spans']:
                    text = span['text'].strip()
                    if not text:
                        continue
                    all_spans.append({
                        'text': text,
                        'bbox': list(span['bbox']),
                        'size': span['size'],
                        'font': span['font'],
                        'is_ru': span['font'] == 'ArialMT',
                    })

        ru_spans = [s for s in all_spans if s['is_ru']]
        eng_spans = [s for s in all_spans if not s['is_ru']]

        scales = []
        for ru in ru_spans:
            rb = ru['bbox']
            ru_width = rb[2] - rb[0]

            # Find best matching English span
            best_eng = None
            best_area = 0
            for eng in eng_spans:
                eb = eng['bbox']
                ox = max(0, min(rb[2], eb[2]) - max(rb[0], eb[0]))
                oy = max(0, min(rb[3], eb[3]) - max(rb[1], eb[1]))
                area = ox * oy
                if area > best_area:
                    best_area = area
                    best_eng = eng

            if not best_eng:
                continue

            eb = best_eng['bbox']
            eng_width = eb[2] - eb[0]
            right_overflow = rb[2] - eb[2]

            if right_overflow <= 3:
                continue

            # Check if overflow actually hits a neighbor
            has_real_overlap = False
            for neighbor in all_spans:
                if neighbor['bbox'] == rb:
                    continue
                nb = neighbor['bbox']
                n_eng_ox = max(0, min(nb[2], eb[2]) - max(nb[0], eb[0]))
                n_eng_oy = max(0, min(nb[3], eb[3]) - max(nb[1], eb[1]))
                if n_eng_ox > 2 and n_eng_oy > 2:
                    continue

                overflow_x0 = eb[2]
                overflow_x1 = rb[2]
                n_ox = max(0, min(overflow_x1, nb[2]) - max(overflow_x0, nb[0]))
                n_oy = max(0, min(rb[3], nb[3]) - max(rb[1], nb[1]))
                if n_ox > 1 and n_oy > 1:
                    has_real_overlap = True
                    break

            if not has_real_overlap:
                continue

            # Calculate scale: fit Russian within English bounds + 2pt margin
            target_width = eng_width + 2
            scale = target_width / ru_width
            if scale > 0.97:
                continue

            scale_pct = scale * 100  # for Tz operator (percentage)
            # Don't compress below 25% - would make text unreadable
            if scale_pct < 25:
                continue

            # Convert fitz position to approximate PDF y coordinate
            # fitz_y (top of bbox) -> PDF y = cropbox_y1 - fitz_y
            fitz_y_center = (rb[1] + rb[3]) / 2
            pdf_y_approx = cropbox_y1 - fitz_y_center

            # Also store fitz x for matching
            fitz_x = rb[0]
            pdf_x_approx = fitz_x + cropbox_x0

            scales.append({
                'pdf_x': pdf_x_approx,
                'pdf_y': pdf_y_approx,
                'fitz_bbox': rb,
                'scale_pct': scale_pct,
                'ru_text': ru['text'],
            })

        if scales:
            page_scales[page_idx] = {
                'scales': scales,
                'cropbox_x0': cropbox_x0,
                'cropbox_y1': cropbox_y1,
            }

    doc.close()

    total_scales = sum(len(v['scales']) for v in page_scales.values())
    print(f"  Found {total_scales} CID streams to scale on {len(page_scales)} pages")

    # Step 2: Modify CID streams with pypdf
    reader = PdfReader(input_path)
    writer = PdfWriter()
    writer.clone_reader_document_root(reader)

    def build_cid_mapping(page):
        resources = page['/Resources'].get_object()
        fonts = resources.get('/Font', {})
        if hasattr(fonts, 'get_object'):
            fonts = fonts.get_object()
        if '/arial' not in fonts:
            return None
        font_obj = fonts['/arial'].get_object()
        if '/ToUnicode' not in font_obj:
            return None
        cmap_data = font_obj['/ToUnicode'].get_object().get_data().decode('latin-1', errors='replace')

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
                    for j in range(e - s + 1):
                        cid_to_unicode[s + j] = chr(u + j)
            if in_bfchar:
                m = re.match(r'<([0-9a-fA-F]+)>\s*<([0-9a-fA-F]+)>', line)
                if m:
                    cid_to_unicode[int(m.group(1), 16)] = chr(int(m.group(2), 16))
        return cid_to_unicode

    total_fixed = 0

    for page_idx in range(len(writer.pages)):
        if page_idx not in page_scales:
            continue

        page = writer.pages[page_idx]
        data = page_scales[page_idx]
        scales = data['scales']
        cropbox_y1 = data['cropbox_y1']
        cropbox_x0 = data['cropbox_x0']

        cid_to_unicode = build_cid_mapping(page)
        if not cid_to_unicode:
            continue

        contents = page.get('/Contents')
        if not contents:
            continue
        obj = contents.get_object()
        try:
            items = list(obj)
        except TypeError:
            continue

        # Parse each CID text stream to find its position
        for stream_idx in range(len(items)):
            stream_ref = items[stream_idx]
            stream_obj = stream_ref.get_object()
            try:
                raw_data = stream_obj.get_data()
            except Exception:
                continue

            stream_text = raw_data.decode('latin-1', errors='replace')

            # Only process CID text streams (contain BT...ET with hex text)
            if 'BT' not in stream_text or '<' not in stream_text:
                continue

            # Already has Tz? Skip
            if ' Tz' in stream_text:
                continue

            # Extract position from Tm line
            # Pattern: a b c d x y Tm
            tm_match = re.search(
                r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+Tm',
                stream_text
            )
            if not tm_match:
                # Try Td pattern: x y Td
                td_match = re.search(r'([\d.]+)\s+([\d.]+)\s+Td', stream_text)
                if not td_match:
                    continue
                stream_pdf_x = float(td_match.group(1))
                stream_pdf_y = float(td_match.group(2))
            else:
                stream_pdf_x = float(tm_match.group(5))
                stream_pdf_y = float(tm_match.group(6))

            # Match this stream position with our scale targets
            best_match = None
            best_dist = 999

            for sc in scales:
                # Compare PDF coordinates
                dx = abs(stream_pdf_x - sc['pdf_x'])
                dy = abs(stream_pdf_y - sc['pdf_y'])
                dist = dx + dy
                if dist < best_dist:
                    best_dist = dist
                    best_match = sc

            if best_match is None or best_dist > 15:
                continue

            # Insert Tz scaling BEFORE the text drawing command (TJ or Tj)
            # Stream format is typically on one line:
            #   ... Tf 0 0 0 RG 0 0 0 rg [<hex>]TJ
            # We need to insert "XX.X Tz " right before [< or <
            scale_pct = best_match['scale_pct']
            tz_str = f'{scale_pct:.1f} Tz '

            # Insert Tz before the first hex text reference: [< or standalone <
            # Pattern: insert before "[<" (array form) or before "<" followed by hex (direct form)
            new_stream_text = re.sub(
                r'(\[<)',
                tz_str + r'\1',
                stream_text,
                count=1
            )
            if new_stream_text == stream_text:
                # Try direct <hex> Tj form
                new_stream_text = re.sub(
                    r'(<[0-9a-fA-F])',
                    tz_str + r'\1',
                    stream_text,
                    count=1
                )

            if new_stream_text != stream_text:
                stream_obj.set_data(new_stream_text.encode('latin-1'))
                total_fixed += 1
                scales.remove(best_match)

    print(f"Phase 1: Applied Tz scaling to {total_fixed} CID text streams")

    with open(output_path, 'wb') as f:
        writer.write(f)


# ======================================================================
# Phase 2: Add missing DIA/DIAMETER translations (fitz)
# ======================================================================

def phase2_add_missing_dia(input_path, output_path):
    doc = fitz.open(input_path)
    font = fitz.Font(fontfile=FONT_PATH)
    total_added = 0

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        blocks = page.get_text('dict')['blocks']

        eng_spans = []
        ru_spans = []

        for b in blocks:
            if 'lines' not in b:
                continue
            for line in b['lines']:
                for span in line['spans']:
                    text = span['text'].strip()
                    if not text:
                        continue
                    if span['font'] == 'ArialMT':
                        ru_spans.append(span)
                    elif 'Arial' in span['font']:
                        if is_dia_text(text):
                            eng_spans.append(span)

        for eng in eng_spans:
            has_overlay = False
            for ru in ru_spans:
                if rects_overlap(eng['bbox'], ru['bbox'], tol=5):
                    has_overlay = True
                    break

            if not has_overlay:
                eng_text = eng['text'].strip()
                ru_text = translate_dia_text(eng_text)
                bbox = eng['bbox']

                eng_rect = fitz.Rect(bbox[0] - 0.3, bbox[1], bbox[2] + 0.5, bbox[3] + 0.5)
                page.draw_rect(eng_rect, color=None, fill=(1, 1, 1))

                fontsize = eng['size'] * 0.85
                tw = fitz.TextWriter(page.rect)
                ascent = font.ascender * fontsize / 1000 if font.ascender else fontsize * 0.75
                baseline_y = bbox[1] + ascent + 1

                try:
                    tw.append(fitz.Point(bbox[0], baseline_y), ru_text,
                              font=font, fontsize=fontsize)
                    tw.write_text(page)
                    total_added += 1
                    print(f"  Page {page_idx + 1}: Added '{ru_text}' for '{eng_text}'")
                except Exception as e:
                    print(f"  Page {page_idx + 1}: ERROR: {e}")

    print(f"Phase 2: Added {total_added} missing DIA translations")
    doc.save(output_path)
    doc.close()


# ======================================================================
# Phase 3: Fix grammar in CID streams (ДО ТОЧКА -> ДО ТОЧКИ)
# ======================================================================

def phase3_fix_grammar(input_path, output_path):
    def build_cid_mapping(page):
        resources = page['/Resources'].get_object()
        fonts = resources.get('/Font', {})
        if hasattr(fonts, 'get_object'):
            fonts = fonts.get_object()
        if '/arial' not in fonts:
            return None, None
        font_obj = fonts['/arial'].get_object()
        if '/ToUnicode' not in font_obj:
            return None, None
        cmap_data = font_obj['/ToUnicode'].get_object().get_data().decode('latin-1', errors='replace')

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
                    for j in range(e - s + 1):
                        cid_to_unicode[s + j] = chr(u + j)
            if in_bfchar:
                m = re.match(r'<([0-9a-fA-F]+)>\s*<([0-9a-fA-F]+)>', line)
                if m:
                    cid_to_unicode[int(m.group(1), 16)] = chr(int(m.group(2), 16))

        unicode_to_cid = {}
        for cid, uc in cid_to_unicode.items():
            if uc not in unicode_to_cid:
                unicode_to_cid[uc] = cid

        return cid_to_unicode, unicode_to_cid

    def decode_hex_text(hex_str, cid_to_unicode):
        decoded = ''
        for i in range(0, len(hex_str), 4):
            if i + 4 <= len(hex_str):
                cid = int(hex_str[i:i+4], 16)
                decoded += cid_to_unicode.get(cid, '\ufffd')
        return decoded

    def encode_text_to_hex(text, unicode_to_cid):
        hex_parts = []
        for ch in text:
            cid = unicode_to_cid.get(ch)
            if cid is None:
                continue
            hex_parts.append(f'{cid:04x}')
        return ''.join(hex_parts)

    def fix_grammar(text):
        original = text
        normalized = text.replace('\xa0', ' ')
        normalized = normalized.replace('ДО ТОЧКА', 'ДО ТОЧКИ')
        normalized = normalized.replace(' ', '\xa0')
        return normalized, normalized != original

    reader = PdfReader(input_path)
    writer = PdfWriter()
    writer.clone_reader_document_root(reader)

    total_fixes = 0

    for page_idx in range(len(writer.pages)):
        page = writer.pages[page_idx]
        cid_to_unicode, unicode_to_cid = build_cid_mapping(page)
        if not cid_to_unicode:
            continue

        contents = page.get('/Contents')
        if not contents:
            continue
        obj = contents.get_object()
        try:
            items = list(obj)
        except TypeError:
            continue

        for stream_idx in range(1, len(items)):
            stream_ref = items[stream_idx]
            stream_obj = stream_ref.get_object()
            try:
                raw_data = stream_obj.get_data()
            except Exception:
                continue

            stream_text = raw_data.decode('latin-1', errors='replace')
            hex_pattern = re.compile(r'<([0-9a-fA-F]+)>')
            matches = list(hex_pattern.finditer(stream_text))

            if not matches:
                continue

            changed = False
            new_text = stream_text

            for match in reversed(matches):
                hex_str = match.group(1)
                decoded = decode_hex_text(hex_str, cid_to_unicode)
                fixed, was_fixed = fix_grammar(decoded)

                if was_fixed:
                    new_hex = encode_text_to_hex(fixed, unicode_to_cid)
                    start, end = match.start(1), match.end(1)
                    new_text = new_text[:start] + new_hex + new_text[end:]
                    changed = True
                    total_fixes += 1
                    print(f"  Page {page_idx + 1}: Grammar fix applied")

            if changed:
                stream_obj.set_data(new_text.encode('latin-1'))

    print(f"Phase 3: Applied {total_fixes} grammar fixes")
    with open(output_path, 'wb') as f:
        writer.write(f)


# ======================================================================
# Phase 4: Redraw CID overlays that still overflow (fitz)
# ======================================================================

def phase4_fix_remaining_overlaps(input_path, output_path):
    """
    For each page, find ArialMT (CID) spans that extend beyond the
    matching English text bbox and overlap neighbors.
    Cover them with white rect + redraw at smaller size using fitz.
    """
    doc = fitz.open(input_path)
    font = fitz.Font(fontfile=FONT_PATH)
    total_fixed = 0

    # CID decoding helper: read CID → unicode from ArialMT CMap
    def decode_cid_text(text_raw):
        """Try to decode CID garbled text to readable Unicode.
        CID text from ArialMT appears as mojibake; we use the raw bytes."""
        # The text from fitz for CID fonts is already decoded via ToUnicode
        # but shows as replacement chars. Return the text as-is and let
        # translate_dia_text / manual lookup handle it.
        return text_raw

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        blocks = page.get_text('dict')['blocks']

        all_spans = []
        for b in blocks:
            if 'lines' not in b:
                continue
            for line in b['lines']:
                for span in line['spans']:
                    text = span['text'].strip()
                    if not text:
                        continue
                    all_spans.append({
                        'text': text,
                        'bbox': list(span['bbox']),
                        'font': span['font'],
                        'size': span['size'],
                        'is_ru': span['font'] == 'ArialMT',
                    })

        ru_spans = [s for s in all_spans if s['is_ru']]
        eng_spans = [s for s in all_spans if not s['is_ru']]

        fixes_this_page = []

        for ru in ru_spans:
            rb = ru['bbox']
            ru_w = rb[2] - rb[0]

            # Find matching English span
            best_eng = None
            best_area = 0
            for eng in eng_spans:
                eb = eng['bbox']
                ox = max(0, min(rb[2], eb[2]) - max(rb[0], eb[0]))
                oy = max(0, min(rb[3], eb[3]) - max(rb[1], eb[1]))
                area = ox * oy
                if area > best_area:
                    best_area = area
                    best_eng = eng

            if not best_eng:
                continue

            eb = best_eng['bbox']
            eng_w = eb[2] - eb[0]
            right_overflow = rb[2] - eb[2]

            if right_overflow <= 3:
                continue

            # Check if overflow hits a neighbor
            has_overlap = False
            for neighbor in all_spans:
                if neighbor['bbox'] == rb:
                    continue
                nb = neighbor['bbox']
                # Skip neighbors that share the same English origin
                n_ox = max(0, min(nb[2], eb[2]) - max(nb[0], eb[0]))
                n_oy = max(0, min(nb[3], eb[3]) - max(nb[1], eb[1]))
                if n_ox > 2 and n_oy > 2:
                    continue
                # Check overflow region
                ov_x0, ov_x1 = eb[2], rb[2]
                ox2 = max(0, min(ov_x1, nb[2]) - max(ov_x0, nb[0]))
                oy2 = max(0, min(rb[3], nb[3]) - max(rb[1], nb[1]))
                if ox2 > 1 and oy2 > 1:
                    has_overlap = True
                    break

            if not has_overlap:
                continue

            # This CID text overflows and hits a neighbor — needs fix
            # Use the English text bbox as the target area
            ru_text = ru['text']
            eng_text = best_eng['text']

            fixes_this_page.append({
                'ru_bbox': rb,
                'eng_bbox': eb,
                'ru_text': ru_text,
                'eng_text': eng_text,
                'eng_size': best_eng['size'],
                'ru_size': ru['size'],
                'overflow': right_overflow,
            })

        # Apply fixes for this page
        for fix in fixes_this_page:
            rb = fix['ru_bbox']
            eb = fix['eng_bbox']
            ru_text = fix['ru_text']
            eng_size = fix['eng_size']

            # Cover the overflowing CID text with white rectangle
            # Extend beyond both RU and ENG bboxes to catch rendering artifacts
            max_right = max(rb[2], eb[2])
            max_bottom = max(rb[3], eb[3])
            cover_rect = fitz.Rect(
                min(rb[0], eb[0]) - 1,
                min(rb[1], eb[1]) - 1,
                max_right + 8,  # 8pt beyond to catch CID rendering artifacts
                max_bottom + 1
            )
            page.draw_rect(cover_rect, color=None, fill=(1, 1, 1))

            # Now redraw the text within the English bbox width
            target_rect = fitz.Rect(
                eb[0],
                eb[1],
                eb[2] + 2,  # small margin
                eb[3] + 1
            )

            # Calculate font size to fit text in one line within target width
            target_w = target_rect.width
            cur_size = eng_size * 0.85  # start slightly smaller than English

            for factor in [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40]:
                cur_size = eng_size * factor
                text_w = font.text_length(ru_text, fontsize=cur_size)
                if text_w <= target_w:
                    break

            if cur_size < 4.0:
                cur_size = 4.0

            # Draw text using TextWriter for precise positioning
            tw = fitz.TextWriter(page.rect)
            ascender = font.ascender
            if ascender:
                baseline_y = eb[1] + ascender * cur_size / 1000 + 0.5
            else:
                baseline_y = eb[1] + cur_size * 0.75 + 0.5

            try:
                tw.append(fitz.Point(eb[0], baseline_y), ru_text,
                          font=font, fontsize=cur_size)
                tw.write_text(page)
                total_fixed += 1
            except Exception as e:
                print(f"  Page {page_idx + 1}: ERROR redrawing: {e}")

    print(f"Phase 4: Redrawn {total_fixed} overflowing CID text spans")
    doc.save(output_path)
    doc.close()


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 60)
    print("Fix picture_ru_fixed_v2.pdf -> picture_ru_fixed_v4.pdf")
    print("=" * 60)
    print("(Phase 1 Tz-scaling and Phase 4 redraw removed — handled by translate_pdf.py)")

    temp1 = INPUT_PATH.replace('.pdf', '_temp1.pdf')

    # Phase 2: Add missing DIA translations (fitz)
    print("\nPhase 2: Adding missing DIA/DIAMETER translations...")
    phase2_add_missing_dia(INPUT_PATH, temp1)

    # Phase 3: Fix grammar (pypdf)
    print("\nPhase 3: Fixing grammar issues...")
    phase3_fix_grammar(temp1, OUTPUT_PATH)

    # Cleanup
    if os.path.exists(temp1):
        os.remove(temp1)

    print(f"\nDone! Output: {OUTPUT_PATH}")
    print(f"File size: {os.path.getsize(OUTPUT_PATH) / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    main()
