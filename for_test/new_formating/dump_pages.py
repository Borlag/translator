"""Dump raw text from original PDF for specific pages."""
import fitz
import sys

pages = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else list(range(108, 115)) + list(range(127, 135)) + [150, 156, 161, 168, 169, 170, 171, 172, 174, 175, 177, 178, 180, 181, 183]

doc = fitz.open(r'C:\Users\Urdul\Desktop\project\translator\for_test\new_formating\picture.pdf')

for pg_num in pages:
    if pg_num < 1 or pg_num > len(doc):
        continue
    page = doc[pg_num - 1]
    d = page.get_text('dict')
    print(f'\n{"="*60}')
    print(f'PAGE {pg_num}')
    print('='*60)
    for block in d['blocks']:
        if block['type'] != 0:
            continue
        for line in block['lines']:
            text = ''.join(sp['text'] for sp in line['spans'])
            if text.strip():
                print(repr(text))
