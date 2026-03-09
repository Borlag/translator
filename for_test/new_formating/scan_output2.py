"""Scan OUTPUT PDF for remaining English - including 2-letter words and garbled mixed text."""
import fitz, re, sys
from collections import Counter

fname = sys.argv[1] if len(sys.argv) > 1 else 'picture_ru.pdf'
doc = fitz.open(fname)

SKIP_WORDS = {
    'SERMETEL','SERMETAL','SAFRAN','MESSIER','DOWTY','MOLYKOTE','AVIOX','ALOCROM','ARDROX',
    'PCS','IFC','DEF','DLPS','MIL','AMS','NCT','FED','STAN','HPC','IVD','MEK',
    'LOCTITE','GRADE','CAGE','SRM','AECMA','AMM','CMM','IPC','DPL','NDT','ATA',
    'SB','LH','RH','GUIDE','RAD','MAX','MIN','REF','DIA','TYP','INCL','Ltd',
    'LANDING','SYSTEMS','LIMITED','MASTINOX','TYPE','CLASS',
    'PR','AN','MS','NAS','NITRALLOY','UK',
}
# Labels that are OK in Latin: single/double letters, part numbers
LABEL_RE = re.compile(r'^[A-Z]{1,2}$')

def find_issues(text):
    """Find English words (2+ letters) and garbled mixed text."""
    issues = []

    # 1. Find standalone English words (2+ letters)
    for m in re.finditer(r'\b[A-Za-z]{2,}\b', text):
        w = m.group()
        if w.upper() in SKIP_WORDS:
            continue
        if LABEL_RE.match(w):
            continue
        # Skip "to" inside dimension ranges like "0.256 to 0.263in"
        if w.lower() == 'to' and re.search(r'\d\s+to\s+\d', text):
            continue
        # Skip "in" as unit (inches) after numbers
        if w.lower() == 'in' and re.search(r'[\d.)]\s*in[)\s.]', text):
            continue
        # Skip "mm"
        if w.lower() == 'mm':
            continue
        issues.append(('ENG', w))

    # 2. Find garbled mixed Cyrillic+Latin (like "СОПРЯЖЕНИЕED")
    for m in re.finditer(r'[А-Яа-яЁё]+[A-Za-z]+|[A-Za-z]+[А-Яа-яЁё]+', text):
        mixed = m.group()
        # Skip known OK patterns
        if any(skip in mixed.upper() for skip in ['SERMETEL', 'MOLYKOTE', 'PCS']):
            continue
        issues.append(('GARBLED', mixed))

    return issues

all_issues = []
for i, page in enumerate(doc):
    d = page.get_text('dict')
    for block in d['blocks']:
        if block['type'] != 0:
            continue
        for line in block['lines']:
            text = ''.join(sp['text'] for sp in line['spans']).strip()
            if not text:
                continue
            found = find_issues(text)
            if found:
                all_issues.append((i+1, text[:120], found))

# Summarize
eng_words = Counter()
garbled = []
for pg, txt, issues in all_issues:
    for typ, w in issues:
        if typ == 'ENG':
            eng_words[w.upper()] += 1
        else:
            garbled.append((pg, w, txt))

print(f'Total lines with issues: {len(all_issues)}')
print(f'Unique English words: {len(eng_words)}')
print(f'Total English word occurrences: {sum(eng_words.values())}')
print(f'Garbled mixed text instances: {len(garbled)}')
print()

if eng_words:
    print('English words remaining (top 50):')
    for word, count in eng_words.most_common(50):
        print(f'  {word}: {count}x')
    print()

if garbled:
    print('Garbled mixed text:')
    for pg, mixed, txt in garbled:
        print(f'  p{pg}: "{mixed}" in: {txt}')
    print()

print('='*80)
print('All issues by page:')
for pg, txt, issues in all_issues:
    tags = [f'{t}:{w}' for t,w in issues]
    print(f'  p{pg}: {tags}  |  {txt}')
