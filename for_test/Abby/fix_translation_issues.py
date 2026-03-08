#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re, sys, io, os
from docx import Document

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(SCRIPT_DIR, 'merged_stylefix_fixed_gap05_nogrow_stripped_docwide_tbfix5.docx')
OUTPUT = os.path.join(SCRIPT_DIR, 'merged_stylefix_fixed_gap05_nogrow_stripped_docwide_tbfix5_reviewed.docx')
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

counters = {}

def inc(key, n=1):
    counters[key] = counters.get(key, 0) + n

def fix_run_text(text, ctx):
    if not text:
        return text, False
    orig = text

    # 1b. SAFRANSAFRAN
    while 'SAFRANSAFRAN' in text:
        text = text.replace('SAFRANSAFRAN', 'SAFRAN')
        inc('1b_SAFRANSAFRAN')

    # 1c. double dot
    w = 'вставленная'
    if w + '..' in text and w + '...' not in text:
        text = text.replace(w + '..', w + '.')
        inc('1c_double_dot')

    # 2a. glossary
    old_g = 'ОСНОВНАЯ СТОЙКА ШАССИ'
    new_g = 'СТОЙКА ОСНОВНОГО ШАССИ'
    if old_g in text:
        text = text.replace(old_g, new_g)
        inc('2a_stoyka')

    # 2b. proverte -> proverka
    prov_old = 'Проверьте'
    prov_new = 'Проверка'
    if ctx.get('standalone_proverte') and prov_old in text:
        text = text.replace(prov_old, prov_new)
        inc('2b_proverte')

    # 3. imperative forms
    if ctx.get('transmittal_hdr'):
        s_old = 'Снимите и'
        s_new = 'Удалить и'
        if s_old in text:
            text = text.replace(s_old, s_new)
            inc('3a_snimite')
        u_old = 'Уничтожьте страницы'
        u_new = 'Уничтожить страницы'
        if u_old in text:
            text = text.replace(u_old, u_new)
            inc('3b_unichtozhte')
        v_old = 'Вставьте новые/пересмотренные'
        v_new = 'Вставить новые/пересмотренные'
        if v_old in text:
            text = text.replace(v_old, v_new)
            inc('3c_vstavte')

    # 4. UDAR -> OTMETKA
    udar = 'УДАР'
    otmetka = 'ОТМЕТКА'
    if udar in text and ctx.get('mod_udar'):
        text = text.replace(udar, otmetka)
        inc('4_udar')

    # 5. Ogranicheno
    ogr = 'Ограничено'
    na = 'на'
    if ctx.get('ogranicheno'):
        if ogr + ' Safran' in text:
            text = text.replace(ogr + ' Safran', 'Limited ' + na + ' Safran')
            inc('5_ogranicheno')
        elif ogr in text:
            text = text.replace(ogr, 'Limited ' + na)
            inc('5_ogranicheno')

    # 6. Russian month cap
    ru_months = {
        'мар': 'Мар', 'апр': 'Апр',
        'янв': 'Янв', 'ноя': 'Ноя',
        'дек': 'Дек', 'авг': 'Авг',
        'сен': 'Сен', 'окт': 'Окт',
        'июн': 'Июн', 'июл': 'Июл',
        'фев': 'Фев', 'май': 'Май',
    }
    for lo, up in ru_months.items():
        pat = re.compile(r'(?<![а-яА-ЯёЁa-zA-Z])' + re.escape(lo) + r'(\.?\s+\d)')
        n = len(pat.findall(text))
        if n:
            text2 = pat.sub(up + r'\1', text)
            if text2 != text:
                inc('6_ru_month_cap', n)
                text = text2

    # 7. English month -> Russian
    eng_months = {
        'Jan': 'Янв', 'Feb': 'Фев',
        'Mar': 'Мар', 'Apr': 'Апр',
        'May': 'Май', 'Jun': 'Июн',
        'Jul': 'Июл', 'Aug': 'Авг',
        'Sep': 'Сен', 'Oct': 'Окт',
        'Nov': 'Ноя', 'Dec': 'Дек',
    }
    for en, ru in eng_months.items():
        pat = re.compile(r'(?<![a-zA-Z])' + re.escape(en) + r'(\s+\d)')
        n = len(pat.findall(text))
        if n:
            text2 = pat.sub(ru + r'\1', text)
            if text2 != text:
                inc('7_eng_month', n)
                text = text2

    # 8. Figure -> Risunok
    risunok = 'Рисунок'
    if 'Figure ' in text:
        pat = re.compile(r'(?<=[а-яА-ЯёЁ\s:\.])Figure (\d)')
        n = len(pat.findall(text))
        if n:
            text2 = pat.sub(risunok + r' \1', text)
            if text2 != text:
                inc('8_figure', n)
                text = text2

    return text, text != orig


def fix_sSAFRAN(p_el):
    runs = p_el.findall('.//w:r', NS)
    fixes = 0
    for i in range(len(runs) - 1):
        ts_cur = runs[i].findall('.//w:t', NS)
        ts_nxt = runs[i + 1].findall('.//w:t', NS)
        if not ts_cur or not ts_nxt:
            continue
        ct = ts_cur[-1].text or ''
        nt = ts_nxt[0].text or ''
        if ct.endswith('s') and nt.startswith('SAFRAN'):
            if ct == 's':
                ts_cur[-1].text = ''
                fixes += 1
            elif ct.endswith(' s'):
                ts_cur[-1].text = ct[:-2]
                fixes += 1
    if fixes:
        inc('1a_sSAFRAN', fixes)
    return fixes


def cell_context(cell, ti):
    ct = ''
    for t in cell.findall('.//w:t', NS):
        ct += (t.text or '')
    cs = ct.strip()
    ctx = {}
    prov = 'Проверьте'
    if cs == prov or cs == prov + ' ':
        ctx['standalone_proverte'] = True
    if ti in (2, 3, 4, 5):
        snimite = 'Снимите и'
        unichtozhte = 'Уничтожьте'
        vstavte = 'Вставьте'
        if snimite in ct or unichtozhte in ct or vstavte in ct:
            ctx['transmittal_hdr'] = True
    udar = 'УДАР'
    mod = 'МОД'
    num_sign = '№'
    if udar in ct and (mod in ct or num_sign in ct or '#' in ct):
        ctx['mod_udar'] = True
    ogr = 'Ограничено'
    if ogr in ct:
        ctx['ogranicheno'] = True
    return ctx


def fix_SAFRANSAFRAN_cross_run(p_el):
    """Fix SAFRANSAFRAN split across runs, skipping runs without text (e.g. tab runs)."""
    runs = p_el.findall('.//w:r', NS)
    # Build list of (run_index, last_t_element) for runs that have text
    text_runs = []
    for i, r in enumerate(runs):
        ts = r.findall('.//w:t', NS)
        if ts:
            text_runs.append((i, ts))
    fixes = 0
    for idx in range(len(text_runs) - 1):
        _, ts_cur = text_runs[idx]
        _, ts_nxt = text_runs[idx + 1]
        ct = ts_cur[-1].text or ''
        nt = ts_nxt[0].text or ''
        if ct.endswith('SAFRAN') and nt.startswith('SAFRAN'):
            if ct == 'SAFRAN':
                ts_cur[-1].text = ''
                fixes += 1
            elif ct.endswith(' SAFRAN'):
                ts_cur[-1].text = ct[:-len('SAFRAN')]
                fixes += 1
    if fixes:
        inc('1b_SAFRANSAFRAN', fixes)
    return fixes


def fix_Figure_cross_run(p_el):
    """Translate Figure -> Рисунок in Russian-context paragraphs."""
    risunok = 'Рисунок'
    # Check if paragraph contains Russian text
    has_russian = False
    has_figure = False
    for r in p_el.findall('.//w:r', NS):
        for t in r.findall('.//w:t', NS):
            txt = t.text or ''
            if any('а' <= c <= 'я' or 'А' <= c <= 'Я' for c in txt):
                has_russian = True
            if 'Figure ' in txt:
                has_figure = True
    if not has_russian or not has_figure:
        return 0
    fixes = 0
    import re
    pat = re.compile(r'Figure (\d)')
    for r in p_el.findall('.//w:r', NS):
        for t in r.findall('.//w:t', NS):
            txt = t.text or ''
            if 'Figure ' in txt:
                n = len(pat.findall(txt))
                if n:
                    t.text = pat.sub(risunok + r' \1', txt)
                    inc('8_figure', n)
                    fixes += n
    return fixes


def process(root):
    fixes = 0
    # Step 1: fix sSAFRAN cross-run in all paragraphs
    for p in root.findall('.//w:p', NS):
        fixes += fix_sSAFRAN(p)

    # Step 1b: fix SAFRANSAFRAN cross-run
    for p in root.findall('.//w:p', NS):
        fixes += fix_SAFRANSAFRAN_cross_run(p)

    # Step 1c: fix Figure cross-run in Russian paragraphs
    for p in root.findall('.//w:p', NS):
        fixes += fix_Figure_cross_run(p)

    # Step 2: apply context-free fixes to ALL text elements
    for t in root.findall('.//w:t', NS):
        if t.text:
            new, changed = fix_run_text(t.text, {})
            if changed:
                t.text = new
                fixes += 1

    # Step 3: apply context-dependent fixes by traversing tables
    for ti, tbl in enumerate(root.findall('.//w:tbl', NS)):
        for row in tbl.findall('.//w:tr', NS):
            for cell in row.findall('.//w:tc', NS):
                ctx = cell_context(cell, ti)
                if ctx:
                    for t in cell.findall('.//w:t', NS):
                        if t.text:
                            new, changed = fix_run_text(t.text, ctx)
                            if changed:
                                t.text = new
                                fixes += 1
    return fixes

def main():
    print(f'Loading: {INPUT}')
    doc = Document(INPUT)
    print('Loaded OK.')
    print()
    total = 0
    bf = process(doc.element)
    total += bf
    print(f'Body fixes (runs modified): {bf}')
    hf = 0
    for rel in doc.part.rels.values():
        rt = str(rel.reltype)
        if 'header' in rt or 'footer' in rt:
            hf += process(rel.target_part.element)
    total += hf
    print(f'Header/footer fixes (runs modified): {hf}')
    print()
    print('=' * 60)
    print('FIX SUMMARY')
    print('=' * 60)
    labels = [
        ('1a_sSAFRAN', '1a. sSAFRAN cross-run removed'),
        ('1b_SAFRANSAFRAN', '1b. SAFRANSAFRAN -> SAFRAN'),
        ('1c_double_dot', '1c. Double dots fixed'),
        ('2a_stoyka', '2a. STOYKA OSNOVNOGO fixed'),
        ('2b_proverte', '2b. Proverte -> Proverka'),
        ('3a_snimite', '3a. Snimite -> Udalit'),
        ('3b_unichtozhte', '3b. Unichtozhte -> Unichtozhit'),
        ('3c_vstavte', '3c. Vstavte -> Vstavit'),
        ('4_udar', '4.  UDAR -> OTMETKA'),
        ('5_ogranicheno', '5.  Ogranicheno -> Limited'),
        ('6_ru_month_cap', '6.  Russian month capitalized'),
        ('7_eng_month', '7.  English month translated'),
        ('8_figure', '8.  Figure -> Risunok'),
    ]
    for key, label in labels:
        v = counters.get(key, 0)
        print(f'  {label}: {v}')
    print(f'  TOTAL INDIVIDUAL FIXES: {sum(counters.values())}')
    print(f'  TOTAL RUNS MODIFIED:   {total}')
    print()
    print(f'Saving: {OUTPUT}')
    doc.save(OUTPUT)
    print('Done!')


if __name__ == '__main__':
    main()
