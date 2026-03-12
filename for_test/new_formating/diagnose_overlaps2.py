#!/usr/bin/env python3
"""Diagnose REAL overlapping text issues in picture_ru_fixed_v2.pdf.

Focus on:
1. Russian overlay text that is WIDER than English text it replaces
   (could overlap with adjacent content)
2. Russian text overlapping with non-overlapping English text (measurements etc.)
"""

import fitz

INPUT_PATH = r'C:\Users\Urdul\Desktop\project\translator\for_test\new_formating\picture_ru_fixed_v2.pdf'


def analyze():
    doc = fitz.open(INPUT_PATH)

    width_issues = []  # Russian wider than English it overlaces
    ru_ru_issues = []  # Russian-Russian overlaps

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        blocks = page.get_text('dict')['blocks']

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
                        'text': text,
                        'bbox': span['bbox'],
                        'size': span['size'],
                        'font': span['font'],
                    }
                    if span['font'] == 'ArialMT':
                        ru_spans.append(entry)
                    else:
                        eng_spans.append(entry)

        # For each Russian span, find the English span it overlays
        for ru in ru_spans:
            rb = ru['bbox']
            # Find best matching English span (most overlap)
            best_eng = None
            best_overlap = 0
            for eng in eng_spans:
                eb = eng['bbox']
                # Check overlap
                ox = max(0, min(rb[2], eb[2]) - max(rb[0], eb[0]))
                oy = max(0, min(rb[3], eb[3]) - max(rb[1], eb[1]))
                area = ox * oy
                if area > best_overlap:
                    best_overlap = area
                    best_eng = eng

            if best_eng:
                eb = best_eng['bbox']
                ru_width = rb[2] - rb[0]
                eng_width = eb[2] - eb[0]
                # Russian text extends beyond English to the right
                right_overflow = rb[2] - eb[2]
                # Russian text extends beyond English to the left
                left_overflow = eb[0] - rb[0]

                if right_overflow > 3:  # more than 3 points wider on right
                    width_issues.append({
                        'page': page_idx + 1,
                        'ru_text': ru['text'],
                        'eng_text': best_eng['text'],
                        'ru_bbox': rb,
                        'eng_bbox': eb,
                        'right_overflow': right_overflow,
                        'ru_width': ru_width,
                        'eng_width': eng_width,
                    })

        # Check RU-RU overlaps (excluding near-identical positions)
        for i, r1 in enumerate(ru_spans):
            for j, r2 in enumerate(ru_spans):
                if j <= i:
                    continue
                b1, b2 = r1['bbox'], r2['bbox']
                ox = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
                oy = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
                if ox > 2 and oy > 2:
                    ru_ru_issues.append({
                        'page': page_idx + 1,
                        'text1': r1['text'],
                        'bbox1': b1,
                        'text2': r2['text'],
                        'bbox2': b2,
                        'overlap': (ox, oy),
                    })

    doc.close()

    # Print width issues (sorted by overflow)
    print("=" * 70)
    print("RUSSIAN TEXT WIDER THAN ENGLISH (right overflow > 3pt)")
    print("=" * 70)
    width_issues.sort(key=lambda x: -x['right_overflow'])
    for w in width_issues[:80]:  # top 80
        print(f"  Page {w['page']:3d}: overflow={w['right_overflow']:.1f}pt "
              f"ru_w={w['ru_width']:.1f} eng_w={w['eng_width']:.1f}")
        print(f"    RU: '{w['ru_text'][:60]}' at x={w['ru_bbox'][0]:.1f}-{w['ru_bbox'][2]:.1f}")
        print(f"    EN: '{w['eng_text'][:60]}' at x={w['eng_bbox'][0]:.1f}-{w['eng_bbox'][2]:.1f}")
    print(f"\nTotal: {len(width_issues)} Russian spans wider than English")

    # Print RU-RU overlaps
    print("\n" + "=" * 70)
    print("RUSSIAN-RUSSIAN OVERLAPS")
    print("=" * 70)
    for r in ru_ru_issues[:40]:
        print(f"  Page {r['page']:3d}: overlap={r['overlap'][0]:.1f}x{r['overlap'][1]:.1f}")
        print(f"    T1: '{r['text1'][:60]}' at {tuple(round(x,1) for x in r['bbox1'])}")
        print(f"    T2: '{r['text2'][:60]}' at {tuple(round(x,1) for x in r['bbox2'])}")
    print(f"\nTotal: {len(ru_ru_issues)} RU-RU overlap issues")


if __name__ == '__main__':
    analyze()
