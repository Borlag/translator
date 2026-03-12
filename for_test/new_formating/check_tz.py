#!/usr/bin/env python3
"""Check that Tz scaling was actually applied to CID streams."""
from pypdf import PdfReader
import re

path = r'C:\Users\Urdul\Desktop\project\translator\for_test\new_formating\picture_ru_fixed_v3b.pdf'
reader = PdfReader(path)

tz_count = 0
examples = []

for page_idx in range(len(reader.pages)):
    page = reader.pages[page_idx]
    contents = page.get('/Contents')
    if not contents:
        continue
    obj = contents.get_object()
    try:
        items = list(obj)
    except TypeError:
        continue

    for stream_ref in items:
        stream_obj = stream_ref.get_object()
        try:
            raw = stream_obj.get_data()
        except Exception:
            continue
        text = raw.decode('latin-1', errors='replace')
        if ' Tz' in text and 'BT' in text:
            tz_match = re.search(r'([\d.]+)\s+Tz', text)
            if tz_match and float(tz_match.group(1)) < 100:
                tz_count += 1
                if len(examples) < 5:
                    examples.append((page_idx + 1, float(tz_match.group(1)), text[:300]))

print(f"Found {tz_count} CID streams with Tz < 100%")
print()
for pg, tz, txt in examples:
    print(f"Page {pg}: Tz = {tz:.1f}%")
    print(f"  Stream: {txt[:200]}")
    print()
