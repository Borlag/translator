"""Debug translation for specific page - show what's being extracted and translated."""
import fitz
import sys

sys.path.insert(0, r'C:\Users\Urdul\Desktop\project\translator\for_test\new_formating')
from translate_pdf import translate_line, should_skip

pg_num = int(sys.argv[1]) if len(sys.argv) > 1 else 64
fname = sys.argv[2] if len(sys.argv) > 2 else 'picture.pdf'

doc = fitz.open(fname)
page = doc[pg_num - 1]
d = page.get_text('dict')

print(f'Page {pg_num} from {fname}')
print('='*60)
for block in d['blocks']:
    if block['type'] != 0:
        continue
    for line in block['lines']:
        spans = line['spans']
        line_text = ''.join(sp['text'] for sp in spans)
        if not line_text.strip():
            continue
        translated = translate_line(line_text)
        if translated != line_text:
            print(f'  TRANSLATE: {repr(line_text)} -> {repr(translated[:80])}')
        elif should_skip(line_text.strip()):
            pass  # don't show skipped
        else:
            import re
            if re.search(r'[A-Za-z]{3,}', line_text):
                print(f'  UNCHANGED: {repr(line_text)}')
