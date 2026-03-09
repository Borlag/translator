"""Сканирование PDF на оставшийся английский текст."""
import fitz
import re
import sys

fname = sys.argv[1] if len(sys.argv) > 1 else 'picture_ru.pdf'
doc = fitz.open(fname)

ALPHA_RE = re.compile(r'[A-Za-z]{3,}')
CYRILLIC_RE = re.compile(r'[А-Яа-яЁё]')

SKIP_WORDS = {
    'SAFRAN','LANDING','SYSTEMS','LIMITED','MESSIER','DOWTY',
    'MASTINOX','MOLYKOTE','AVIOX','SERMETEL','ALOCROM','ARDROX',
    'ZINC','NICKEL','CADMIUM','CHROMIUM','TYPE','CLASS',
    'PSC','NCT','DEF','DLPS','MIL','AMS','IFC','PCS','FED','STAN',
    'SB','CMM','IPC','DPL','NDT','AMM','ATA','SRM','AECMA',
    'HPC','IVD','MEK','CAGE','GUIDE','LH','RH',
}
OK_RE = re.compile(r'^[A-Z]{1,3}[-/]?\d|^\d|^[A-Z]{1,4}$', re.IGNORECASE)

def eng_words(text):
    return [w for w in ALPHA_RE.findall(text)
            if not OK_RE.match(w) and w.upper() not in SKIP_WORDS]

results = []
for i, page in enumerate(doc):
    d = page.get_text('dict')
    for block in d['blocks']:
        if block['type'] != 0:
            continue
        for line in block['lines']:
            text = ''.join(sp['text'] for sp in line['spans']).strip()
            if not text:
                continue
            eng = eng_words(text)
            if len(eng) >= 2:
                results.append((i+1, eng[:6], text[:120]))

print(f'Строк с 2+ англ. словами: {len(results)}')
print()
# Unique texts
seen = {}
for pg, eng, txt in results:
    key = txt[:80]
    if key not in seen:
        seen[key] = (pg, eng, txt)

print(f'Уникальных: {len(seen)}')
print()
for key, (pg, eng, txt) in list(seen.items()):
    print(f'  p{pg}: [{" ".join(eng)}] -> {txt}')
