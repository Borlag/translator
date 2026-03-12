#!/usr/bin/env python3
"""Find cases where Russian overlay text actually overlaps with NEIGHBORING text
(not the English text it replaced, but OTHER adjacent text)."""

import fitz

INPUT_PATH = r'C:\Users\Urdul\Desktop\project\translator\for_test\new_formating\picture_ru_fixed_v2.pdf'


def analyze():
    doc = fitz.open(INPUT_PATH)
    real_problems = []

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

        for ru in ru_spans:
            rb = ru['bbox']
            ru_text = ru['text']

            # Find the English span this Russian text overlays (best match)
            best_eng = None
            best_area = 0
            for s in all_spans:
                if s['is_ru']:
                    continue
                eb = s['bbox']
                ox = max(0, min(rb[2], eb[2]) - max(rb[0], eb[0]))
                oy = max(0, min(rb[3], eb[3]) - max(rb[1], eb[1]))
                area = ox * oy
                if area > best_area:
                    best_area = area
                    best_eng = s

            if not best_eng:
                continue

            # Now check: does this Russian text overlap with ANY other span
            # that is NOT the English span it replaced?
            for neighbor in all_spans:
                if neighbor is best_eng:
                    continue
                if neighbor['is_ru'] and neighbor['bbox'] == ru['bbox']:
                    continue  # skip self

                nb = neighbor['bbox']
                # Check overlap
                ox = max(0, min(rb[2], nb[2]) - max(rb[0], nb[0]))
                oy = max(0, min(rb[3], nb[3]) - max(rb[1], nb[1]))

                if ox > 2 and oy > 2:
                    # Check if this overlap is because neighbor is ALSO an overlay
                    # for the same English text (skip those)
                    eb = best_eng['bbox']
                    nb_eng_ox = max(0, min(nb[2], eb[2]) - max(nb[0], eb[0]))
                    nb_eng_oy = max(0, min(nb[3], eb[3]) - max(nb[1], eb[1]))
                    if nb_eng_ox > 2 and nb_eng_oy > 2:
                        continue  # neighbor overlaps with same English, skip

                    real_problems.append({
                        'page': page_idx + 1,
                        'ru_text': ru_text,
                        'ru_bbox': rb,
                        'neighbor_text': neighbor['text'],
                        'neighbor_bbox': nb,
                        'neighbor_is_ru': neighbor['is_ru'],
                        'overlap': (ox, oy),
                        'eng_text': best_eng['text'],
                    })

    doc.close()

    # Deduplicate and sort by significance
    seen = set()
    unique = []
    for p in real_problems:
        key = (p['page'], p['ru_text'][:30], p['neighbor_text'][:30])
        if key not in seen:
            seen.add(key)
            unique.append(p)

    unique.sort(key=lambda x: -(x['overlap'][0] * x['overlap'][1]))

    print(f"Found {len(unique)} real overlap problems:\n")
    for p in unique[:100]:
        area = p['overlap'][0] * p['overlap'][1]
        nbtype = 'RU' if p['neighbor_is_ru'] else 'EN'
        print(f"  Page {p['page']:3d}: area={area:.0f} ({p['overlap'][0]:.1f}x{p['overlap'][1]:.1f})")
        print(f"    RU overlay: '{p['ru_text'][:50]}' at x={p['ru_bbox'][0]:.0f}-{p['ru_bbox'][2]:.0f}")
        print(f"    Overlaps [{nbtype}]: '{p['neighbor_text'][:50]}' at x={p['neighbor_bbox'][0]:.0f}-{p['neighbor_bbox'][2]:.0f}")
        print(f"    (was overlay for EN: '{p['eng_text'][:40]}')")
        print()


if __name__ == '__main__':
    analyze()
