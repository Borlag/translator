"""Dump raw text with repr() from both original and output PDFs for specific pages."""
import fitz
import sys

pages = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else [64, 65, 66, 84, 149, 163, 164, 169]
fname = sys.argv[1] if len(sys.argv) > 1 else 'picture.pdf'

doc = fitz.open(fname)

for pg_num in pages:
    if pg_num < 1 or pg_num > len(doc):
        continue
    page = doc[pg_num - 1]
    d = page.get_text('dict')
    print(f'\n{"="*60}')
    print(f'PAGE {pg_num} from {fname}')
    print('='*60)
    for block in d['blocks']:
        if block['type'] != 0:
            continue
        for line in block['lines']:
            text = ''.join(sp['text'] for sp in line['spans'])
            if text.strip():
                print(repr(text))
