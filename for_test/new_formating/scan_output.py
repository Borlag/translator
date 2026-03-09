"""Scan the OUTPUT translated PDF for any remaining English words."""
import fitz, re, sys

fname = sys.argv[1] if len(sys.argv) > 1 else 'picture_ru.pdf'
doc = fitz.open(fname)

SKIP_WORDS = {
    'SERMETEL','SERMETAL','SAFRAN','MESSIER','DOWTY','MOLYKOTE','AVIOX','ALOCROM','ARDROX',
    'PCS','IFC','DEF','DLPS','MIL','AMS','NCT','FED','STAN','HPC','IVD','MEK',
    'LOCTITE','GRADE','CAGE','SRM','AECMA','AMM','CMM','IPC','DPL','NDT','ATA',
    'SB','LH','RH','GUIDE','RAD','MAX','MIN','REF','DIA','TYP','INCL','Ltd',
    'LANDING','SYSTEMS','LIMITED','MASTINOX','TYPE','CLASS',
    'PR','AN','MS','NAS','NITRALLOY',
}
# Skip patterns: part numbers, spec codes, etc.
SKIP_RE = re.compile(r'^[A-Z]{1,4}[-/]?\d|^\d|^[A-Z]{1,3}$|^[a-z]{1,2}$', re.IGNORECASE)

def find_english(text):
    """Find English words (3+ letters) that shouldn't be there."""
    words = re.findall(r'[A-Za-z]{3,}', text)
    result = []
    for w in words:
        if w.upper() in SKIP_WORDS:
            continue
        if SKIP_RE.match(w):
            continue
        # Skip if it looks like a unit or code
        if re.match(r'^[a-z]{1,2}$', w):
            continue
        result.append(w)
    return result

issues = []
for i, page in enumerate(doc):
    d = page.get_text('dict')
    for block in d['blocks']:
        if block['type'] != 0:
            continue
        for line in block['lines']:
            text = ''.join(sp['text'] for sp in line['spans']).strip()
            if not text:
                continue
            eng = find_english(text)
            if eng:
                issues.append((i+1, text[:120], eng))

print(f'Lines with remaining English words: {len(issues)}')

# Count unique words
from collections import Counter
word_counts = Counter()
for pg, txt, words in issues:
    for w in words:
        word_counts[w.upper()] += 1

print(f'Unique English words: {len(word_counts)}')
print(f'Total occurrences: {sum(word_counts.values())}')
print()
print('Top 50 most frequent:')
for word, count in word_counts.most_common(50):
    print(f'  {word}: {count}x')

print()
print('='*80)
print('All issues by page:')
seen_pages = set()
for pg, txt, words in issues:
    print(f'  p{pg}: {words}  |  {txt}')
