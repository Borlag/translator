#!/usr/bin/env python3
"""Quick check: compare overlap counts between v2 and v3b."""
import fitz

def count_real_overlaps(path):
    doc = fitz.open(path)
    total = 0
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
                        'bbox': list(span['bbox']),
                        'font': span['font'],
                        'text': text,
                        'is_ru': span['font'] == 'ArialMT',
                    })

        ru_spans = [s for s in all_spans if s['is_ru']]
        eng_spans = [s for s in all_spans if not s['is_ru']]

        for ru in ru_spans:
            rb = ru['bbox']
            # Find matching English
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
            right_overflow = rb[2] - eb[2]
            if right_overflow <= 3:
                continue

            # Check neighbor overlap
            for neighbor in all_spans:
                if neighbor['bbox'] == rb:
                    continue
                nb = neighbor['bbox']
                eng_ox = max(0, min(nb[2], eb[2]) - max(nb[0], eb[0]))
                eng_oy = max(0, min(nb[3], eb[3]) - max(nb[1], eb[1]))
                if eng_ox > 2 and eng_oy > 2:
                    continue
                overflow_x0 = eb[2]
                overflow_x1 = rb[2]
                n_ox = max(0, min(overflow_x1, nb[2]) - max(overflow_x0, nb[0]))
                n_oy = max(0, min(rb[3], nb[3]) - max(rb[1], nb[1]))
                if n_ox > 1 and n_oy > 1:
                    total += 1
                    break

    doc.close()
    return total


v2 = r'C:\Users\Urdul\Desktop\project\translator\for_test\new_formating\picture_ru_fixed_v2.pdf'
v3b = r'C:\Users\Urdul\Desktop\project\translator\for_test\new_formating\picture_ru_fixed_v3b.pdf'

print("Counting overlaps in v2...")
c2 = count_real_overlaps(v2)
print(f"  v2: {c2} real overlaps")

print("Counting overlaps in v3b...")
c3 = count_real_overlaps(v3b)
print(f"  v3b: {c3} real overlaps")

print(f"\nImprovement: {c2} -> {c3} ({c2-c3} fixed, {(c2-c3)/c2*100:.0f}% reduction)")
