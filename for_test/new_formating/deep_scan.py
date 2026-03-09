"""Deep scan for remaining English in translated output."""
import fitz, re, sys
sys.path.insert(0, '.')
from translate_pdf import translate_line, should_skip

doc = fitz.open('picture.pdf')

SKIP_WORDS = {
    'SERMETEL','SAFRAN','MESSIER','DOWTY','MOLYKOTE','AVIOX','ALOCROM','ARDROX',
    'PCS','IFC','DEF','DLPS','MIL','AMS','NCT','FED','STAN','HPC','IVD','MEK',
    'LOCTITE','GRADE','CAGE','SRM','AECMA','AMM','CMM','IPC','DPL','NDT','ATA',
    'SB','LH','RH','GUIDE','RAD','MAX','MIN','REF','DIA','TYP','INCL','Ltd',
    'LANDING','SYSTEMS','LIMITED','MASTINOX','TYPE','CLASS','SERMETAL',
}
# lowercase ok words (not needing translation in context)
SKIP_LOWER = {'for','and','or','to','in','of','the','on','at','is','it','an','as','by','mm'}

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
            translated = translate_line(text)
            if should_skip(text.strip()):
                continue

            # Find remaining English words in translated text
            eng_remaining = re.findall(r'[A-Za-z]{2,}', translated)
            remaining = [w for w in eng_remaining
                        if w.upper() not in SKIP_WORDS
                        and w.lower() not in SKIP_LOWER
                        and not re.match(r'^[A-Z]{1,3}$', w)  # single letter labels
                        and not re.match(r'^\d', w)]

            if remaining:
                label = 'PARTIAL' if translated != text else 'UNCHANGED'
                issues.append((i+1, text[:90], translated[:90], remaining, label))

print(f'Total lines with remaining English: {len(issues)}')
print()

# Group by unique remaining words
from collections import Counter
word_counts = Counter()
for pg, orig, trans, words, label in issues:
    for w in words:
        word_counts[w.upper()] += 1

print(f'Unique English words remaining: {len(word_counts)}')
print('Top 40 most frequent:')
for word, count in word_counts.most_common(40):
    print(f'  {word}: {count}x')

print()
print('='*80)
print('All issues:')
for pg, orig, trans, words, label in issues:
    print(f'  p{pg} [{label}]: {words}')
    print(f'    ORIG: {orig}')
    print(f'    TRANS: {trans}')
    print()
