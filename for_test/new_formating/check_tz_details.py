#!/usr/bin/env python3
"""Check Tz scaling values - look for very aggressive ones that are likely wrong."""
from pypdf import PdfReader
import re

path = r'C:\Users\Urdul\Desktop\project\translator\for_test\new_formating\picture_ru_fixed_v3b.pdf'
reader = PdfReader(path)

tz_values = []

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
        tz_match = re.search(r'([\d.]+)\s+Tz\s+\[<', text)
        if tz_match:
            tz = float(tz_match.group(1))
            if tz < 100:
                tz_values.append((page_idx + 1, tz))

# Distribution
print(f"Total streams with Tz scaling: {len(tz_values)}")
print(f"\nDistribution:")
ranges = [(0, 30), (30, 50), (50, 70), (70, 80), (80, 90), (90, 100)]
for lo, hi in ranges:
    count = sum(1 for _, tz in tz_values if lo <= tz < hi)
    print(f"  {lo}-{hi}%: {count} streams")

print(f"\nMost aggressive (Tz < 50%):")
for pg, tz in sorted(tz_values, key=lambda x: x[1])[:20]:
    print(f"  Page {pg}: Tz = {tz:.1f}%")
