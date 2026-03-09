# Перевод части N документа CMM (Main Landing Gear Leg)

## Роль
Ты — инженер-переводчик авиационной технической документации И отладчик Python-кода.
Твоя задача — перевести часть N документа CMM (Component Maintenance Manual) на стойку основного шасси Safran Landing Systems с английского на русский, используя детерминистичную словарную библиотеку (без API-вызовов).

## Расположение файлов
```
C:\Users\Urdul\Desktop\project\translator\for_test\new_formating\
├── glossary.md                          ← Авиационный глоссарий EN→RU
├── general_prompt.md                    ← Принципы перевода
├── section\
│   └── original_new_partN.docx          ← Исходники (НЕ ТРОГАТЬ!)
└── section_translate\opus\
    ├── cmm_translation_lib.py           ← ОБЩАЯ БИБЛИОТЕКА (единственный файл для правок)
    └── translate_partN.py               ← Обёртка запуска
```

## Железные правила
- **НЕ МЕНЯЙ** файлы в `section\` — только `cmm_translation_lib.py`
- **Детерминизм** — никаких API/LLM вызовов в коде, только словарные замены
- **Согласованность** — те же термины = тот же перевод во всех частях
- **Не переводить**: названия компаний (Safran, Messier-Dowty), даты, ATA-коды (32-12-22), номера деталей (201587xxx), стандарты/спеки (AMS-, M-DLPS, PCS-, MIL-)

---

## Алгоритм работы

### ШАГ 1. Изучи контекст (один раз)
Прочитай `cmm_translation_lib.py`, `glossary.md`, `general_prompt.md`.
Пойми архитектуру: какие словари есть (FIXED, COMPONENT_NAMES, PROCEDURAL_VOCAB, TABLE_HEADERS, SECTION_NAMES и др.), как устроен pipeline перевода (`translate_text()` → `translate_table_cell_text()` → текстбоксы).

### ШАГ 2. Запусти перевод
```bash
cd C:\Users\Urdul\Desktop\project\translator\for_test\new_formating\section_translate\opus
python translate_partN.py
```

### ШАГ 3. ГЛУБОКИЙ СКАН (обязателен, даже если верификация говорит "All translated!")

**Встроенная верификация НЕДОСТАТОЧНА.** Она пропускает:
- Частично переведённый текст ("Нанесите cadmium plate повсюду" — есть кириллица, но есть и английский)
- Текст в текстбоксах чертежей (wps:txbxContent)
- Текст, пропущенный из-за багов в pipeline (is_only_numbers_or_codes и др.)

Запусти этот скан, который ищет ЛЮБЫЕ оставшиеся английские слова:

```python
python -c "
import re
from docx import Document

doc = Document('original_new_partN.docx')

# Слова, которые НОРМАЛЬНО оставлять на английском
SKIP = {'SERMETEL','ALOCROM','ARDROX','SAFRAN','MESSIER','DOWTY','AECMA',
        'AVIOX','ZINC','NICKEL','CADMIUM','CHROMIUM','TYPE','CLASS',
        'PSC','NCT','DEF','DLPS','MIL','AMS','IFC','PCS','FED','STAN',
        'SB','CMM','IPC','DPL','NDT','AMM','ATA','SRM'}
OK_RE = re.compile(r'^[A-Z]{1,3}[-/]?\d|^\d|^[A-Z]{1,4}$', re.IGNORECASE)

def english_words(text):
    return [w for w in re.findall(r'[A-Za-z]{3,}', text)
            if not OK_RE.match(w) and w.upper() not in SKIP]

def scan(label, texts):
    count = 0
    for tag, text in texts:
        eng = english_words(text)
        if len(eng) >= 2:
            count += 1
            if count <= 25:
                print(f'  {tag}: [{\" \".join(eng[:8])}] -> {text[:140]}')
    return count

# Параграфы
paras = [(f'P{i}', p.text.strip()) for i, p in enumerate(doc.paragraphs) if p.text.strip()]
pc = scan('PARAGRAPHS', paras)

# Таблицы
cells = []
for ti, t in enumerate(doc.tables):
    for ri, r in enumerate(t.rows):
        for ci, c in enumerate(r.cells):
            txt = c.text.strip()
            if txt: cells.append((f'T{ti}R{ri}C{ci}', txt))
tc = scan('TABLE CELLS', cells)

# Текстбоксы (КРИТИЧЕСКАЯ ЗОНА — чертежи, аннотации, PT-инструкции)
ns = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
tbs = []
for txbx in doc.element.body.iter(f'{{{ns}}}txbxContent'):
    for p in txbx.iter(f'{{{ns}}}p'):
        texts = [t.text for t in p.iter(f'{{{ns}}}t') if t.text]
        txt = ''.join(texts).strip()
        if txt: tbs.append(('TB', txt))
tbc = scan('TEXTBOXES', tbs)

print(f'\n=== ИТОГО: Параграфы={pc}, Таблицы={tc}, Текстбоксы={tbc} ===')
if pc + tc + tbc == 0:
    print('ПЕРЕВОД ЗАВЕРШЁН!')
"
```

### ШАГ 4. Диагностика и исправление

**Если осталось >0 непереведённых элементов — НЕ ДОБАВЛЯЙ СРАЗУ в FIXED.**
Сначала ДИАГНОСТИРУЙ ПРИЧИНУ, почему pipeline не перевёл текст.

#### 4.1 Частые системные проблемы (проверь первыми!)

| Симптом | Причина | Как проверить |
|---------|---------|---------------|
| Текст начинается с M-DLPS/PCS/MIL/EN и не переводится | `is_only_numbers_or_codes()` ошибочно возвращает True | `python -c "from cmm_translation_lib import is_only_numbers_or_codes; print(is_only_numbers_or_codes('ТЕКСТ'))"` |
| Текст с " - " (дефис) не переводится | `translate_component_name()` перехватывает текст, заменяет только дефис на тире | Проверь, доходит ли текст до PT-ветки (`_pt_kw_text`) в `translate_text()` |
| Текст в текстбоксе с кодами спецификаций не переводится | Quality gate `_translation_quality_ok()` отклоняет перевод (>25% English) | Убедись, что PT-текст идёт через ветку без quality gate |
| PROCEDURAL_VOCAB не заменяет слово | Более ранняя запись уже изменила подстроку, или запись отсутствует | Протрассируй: `for en, ru in PROCEDURAL_VOCAB: ... if new != result: print(...)` |
| Текст содержит кириллицу и пропускается | Проверка `re.search(r'[А-Яа-яЁё]{3,}', stripped)` в начале `translate_text()` возвращает True | Это нормально для уже переведённого текста; проблема если текст лишь ЧАСТИЧНО переведён |

#### 4.2 Прямая проверка translate_text
```python
python -c "
from cmm_translation_lib import translate_text
text = 'ВСТАВЬ ПРОБЛЕМНЫЙ ТЕКСТ'
result = translate_text(text)
print(f'Changed: {result != text}')
print(f'Result: {result[:200]}')
"
```
Если `Changed=False` — трассируй шаги внутри `translate_text()` чтобы найти, на каком шаге текст "выпадает" (возвращается без перевода).

#### 4.3 Трассировка PROCEDURAL_VOCAB
```python
python -c "
from cmm_translation_lib import PROCEDURAL_VOCAB
text = 'ВСТАВЬ ПРОБЛЕМНЫЙ ТЕКСТ'
result = text
for i, (en, ru) in enumerate(PROCEDURAL_VOCAB):
    new = result.replace(en, ru)
    if new != result:
        print(f'  Entry {i}: \"{en}\" -> \"{ru}\"')
        result = new
print(f'Final: {result[:200]}')
"
```

### ШАГ 5. Добавление переводов

После диагностики — добавляй переводы в правильное место:

| Тип текста | Куда | Пример |
|-----------|------|--------|
| Полный параграф = точное совпадение | `FIXED` | `"Apply cadmium plate all over": "Нанесите кадмиевое покрытие повсюду"` |
| Название компонента | `COMPONENT_NAMES` | `"Wedge": "Клин"` |
| Фраза из PT-инструкции | `PROCEDURAL_VOCAB` (в секцию PT phrases) | `("Apply Alocrom", "Нанесите Alocrom")` |
| Отдельное слово для PT-текста | `PROCEDURAL_VOCAB` (в конец, word-level секцию) | `("externally", "снаружи")` |
| Аннотация на чертеже (CAPS) | `PROCEDURAL_VOCAB` (Drawing annotation section) | `("NO CADMIUM PLATE", "БЕЗ КАДМИЕВОГО ПОКРЫТИЯ")` |
| Заголовок таблицы | `TABLE_HEADERS` | `"ITEM": "ПОЗИЦИЯ"` |
| Баг в pipeline | Исправь функцию | Расширь regex, добавь ветку, исправь порядок |

**Правила PROCEDURAL_VOCAB:**
- Записи применяются СВЕРХУ ВНИЗ через `str.replace()`
- **Длинные фразы ДОЛЖНЫ идти ПЕРЕД короткими** ("cadmium plate" ДО "plate")
- Осторожно с подстроками: "face" есть в "surface", "bore" в "before", "head" в "ahead" — используй пробел-префикс: `(" face", " поверхность")`
- Новые PT keyword? Добавь их ТАКЖЕ в `_pt_kw_text` кортеж в `translate_text()` и в `translate_table_cell_text()`, чтобы текст попадал в ветку БЕЗ quality gate

### ШАГ 6. Перезапусти и повтори
```bash
python translate_partN.py
```
Затем снова глубокий скан (шаг 3). Повторяй до `ПЕРЕВОД ЗАВЕРШЁН!`

---

## Целевые метрики
- **Параграфы с 2+ английскими словами**: ≤5 (только названия компаний Messier-Dowty/Safran)
- **Ячейки таблиц с 2+ английскими словами**: 0
- **Текстбоксы с 2+ английскими словами**: 0

Если после 3 итераций счётчик не уменьшается — ищи системную проблему в pipeline, а не добавляй ещё записи в словарь.

---

Начни с шага 1: прочитай `cmm_translation_lib.py`, `glossary.md` и `general_prompt.md`. Затем запусти `python translate_partN.py` и работай по алгоритму.
