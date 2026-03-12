#!/usr/bin/env python3
"""Diagnose overlapping text issues in picture_ru_fixed_v2.pdf.

For each page, find:
1. Russian overlay text that overlaps with other Russian overlay text
2. Russian overlay text that overlaps with English text (measurement labels etc.)
3. White rects in overlay streams that may be too small or too large
"""

import fitz
import re

INPUT_PATH = r'C:\Users\Urdul\Desktop\project\translator\for_test\new_formating\picture_ru_fixed_v2.pdf'


def analyze_overlaps():
    doc = fitz.open(INPUT_PATH)

    problems = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        blocks = page.get_text('dict')['blocks']

        all_spans = []
        ru_spans = []
        eng_spans = []

        for b in blocks:
            if 'lines' not in b:
                continue
            for line in b['lines']:
                for span in line['spans']:
                    text = span['text'].strip()
                    if not text:
                        continue
                    entry = {
                        'text': span['text'],
                        'bbox': span['bbox'],
                        'size': span['size'],
                        'font': span['font'],
                    }
                    all_spans.append(entry)
                    if span['font'] == 'ArialMT':
                        ru_spans.append(entry)
                    else:
                        eng_spans.append(entry)

        # Check Russian-Russian overlaps
        for i, r1 in enumerate(ru_spans):
            for j, r2 in enumerate(ru_spans):
                if j <= i:
                    continue
                b1, b2 = r1['bbox'], r2['bbox']
                # Check horizontal overlap
                x_overlap = b1[0] < b2[2] and b2[0] < b1[2]
                y_overlap = b1[1] < b2[3] and b2[1] < b1[3]
                if x_overlap and y_overlap:
                    # Calculate overlap area
                    ox = min(b1[2], b2[2]) - max(b1[0], b2[0])
                    oy = min(b1[3], b2[3]) - max(b1[1], b2[1])
                    if ox > 1 and oy > 1:  # meaningful overlap
                        problems.append({
                            'page': page_idx + 1,
                            'type': 'RU-RU overlap',
                            'text1': r1['text'].strip(),
                            'bbox1': r1['bbox'],
                            'text2': r2['text'].strip(),
                            'bbox2': r2['bbox'],
                            'overlap': (ox, oy),
                        })

        # Check Russian-English overlaps (Russian text overlapping English measurements)
        for ru in ru_spans:
            for eng in eng_spans:
                b1, b2 = ru['bbox'], eng['bbox']
                x_overlap = b1[0] < b2[2] and b2[0] < b1[2]
                y_overlap = b1[1] < b2[3] and b2[1] < b1[3]
                if x_overlap and y_overlap:
                    ox = min(b1[2], b2[2]) - max(b1[0], b2[0])
                    oy = min(b1[3], b2[3]) - max(b1[1], b2[1])
                    if ox > 1 and oy > 1:
                        problems.append({
                            'page': page_idx + 1,
                            'type': 'RU-ENG overlap',
                            'text1': ru['text'].strip(),
                            'bbox1': ru['bbox'],
                            'text2': eng['text'].strip(),
                            'bbox2': eng['bbox'],
                            'overlap': (ox, oy),
                        })

    doc.close()

    # Print results
    print(f"Found {len(problems)} overlap issues:\n")

    # Group by page
    by_page = {}
    for p in problems:
        pg = p['page']
        if pg not in by_page:
            by_page[pg] = []
        by_page[pg].append(p)

    for pg in sorted(by_page.keys()):
        print(f"=== Page {pg} ===")
        for p in by_page[pg]:
            print(f"  [{p['type']}] overlap={p['overlap'][0]:.1f}x{p['overlap'][1]:.1f}")
            print(f"    Text1: '{p['text1']}' at {tuple(round(x,1) for x in p['bbox1'])}")
            print(f"    Text2: '{p['text2']}' at {tuple(round(x,1) for x in p['bbox2'])}")
        print()

    return problems


if __name__ == '__main__':
    problems = analyze_overlaps()
    print(f"\nTotal: {len(problems)} problems on {len(set(p['page'] for p in problems))} pages")
