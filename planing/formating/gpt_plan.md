Ниже — разбор, почему именно у тебя «плывёт» форматирование в переведённом DOCX 

merged_stylefix

, и план, как довести пайплайн до состояния «файл можно отдавать заказчику без ручной вёрстки».

1) Что реально ломает “готовность к заказчику” в твоём кейсе
A. Документ типично “ABBYY-DOCX” (layout-rigid)

В таких DOCX текст почти всегда сидит в тысячах текстбоксов/фреймов и/или таблиц, а ABBYY часто выставляет жёсткие ограничения:

noAutofit у текстбоксов ⇒ Word не увеличивает область под текст и обрезает RU.

w:hRule="exact" у строк таблиц / фреймов ⇒ высота фиксирована ⇒ снова обрезка.

w:spacing w:lineRule="exact" ⇒ “жёсткая” межстрочка ⇒ обрезка на кириллице.

Т.е. тут недостаточно “сохранить стили” — надо снять механические ограничения, иначе RU физически не помещается.

B. В проекте фиксы есть, но дефолтами они выключены

Claude-план правильно указывает, что layout_check/layout_auto_fix/abbyy_profile/translate_enable_formatting_fixes часто стоят в false/off 

claude_plan

 

claude_plan

.
Это подтверждается дефолтами PipelineConfig: mode="reflow", translate_enable_formatting_fixes=False, abbyy_profile="off", layout_check=False, layout_auto_fix=False 

config

.
=> В результате пайплайн может честно “перевести”, но вообще не пытаться чинить переполнение.

C. Write-back пересобирает runs (не всегда критично, но влияет)

tagged_to_runs() документированно удаляет существующие runs и создаёт новые через paragraph.add_run() 

tagging

 

tagging

.
В обычных DOCX это ок, но в ABBYY-доках иногда ломает “микро-геометрию” (особенно в узких контейнерах).

2) План “готовый к заказчику” (без магии, с контрольными точками)
Фаза 0 — Быстрый “победный” пресет (1–2 часа)

Цель: включить уже существующие механизмы и получить резкое падение числа обрезок.

Для уже переведённого файла запускай postformat:
CLI это поддерживает: docxru postformat --input ... --output ... --config ... --abbyy-profile ... --mode ... 

cli

Минимальный “боевой” конфиг под ABBYY:

abbyy_profile: full (снимает ABBYY-жёсткости + включает autofit текстбоксов)

layout_check: true

layout_auto_fix: true

layout_auto_fix_passes: 2..4

mode: com (если есть Word на машине/сервере) + включить расширение текстбоксов

Почему abbyy_profile=full важно: в коде явно написано, что full делает set_textbox_autofit() + нормализацию, а aggressive/safe — более мягкие режимы 

oxml_table_fix

.

Если ты делаешь перевод (translate), то включай translate_enable_formatting_fixes: true, иначе ABBYY/layout-проходы могут не запускаться в translate-пайплайне (Claude это тоже отметил) 

claude_plan

 

config

.

Фаза 1 — Пресеты + авто-детект (1–2 дня)

Это прям “как продукт”.

1A) Ввести formatting_preset + таблицу пресетов (как в Claude-плане) 

claude_plan


Пресет должен атомарно выставлять:

abbyy_profile

layout_check/layout_auto_fix/layout_auto_fix_passes

mode

com_expand_overflowing_shapes и лимиты

глобальный font_shrink_* (обычно держать 0 и включать только как “последний шанс”)

1B) Авто-детект “ABBYY vs normal DOCX”
Идея Claude: проверять creator/last_modified_by и долю w:framePr/noAutofit 

claude_plan

.
Практически: на входе считать 3–4 счётчика по word/document.xml и выбирать пресет автоматически.

Фаза 2 — Детерминированное снятие “железных ограничений” (2–4 дня)

Даже с abbyy_profile=full лучше усилить “нормализатор” под реальные ABBYY-файлы.

2A) Расширить ABBYY-normalization:

для таблиц: убирать/релаксить всё hRule="exact" (строки и фреймы) системно

для параграфов: lineRule="exact" -> atLeast (уже есть) 

oxml_table_fix

для текстбоксов: не только <a:noAutofit/>, но и вариации в VML/старых shape-схемах (ABBYY часто мешает namespaces)

2B) Правильная привязка размеров контейнера к сегменту
Сейчас layout-детектор берёт wp:extent “где-то в предках”, что в ABBYY-лесу может цеплять не тот контейнер.
Надёжнее: прямо в collect_segments() при обходе w:txbxContent 

docx_reader

 сохранить в seg.context:

textbox_id (порядковый/хеш xpath)

extent (w,h) конкретного shape
Тогда layout_check использует точный контейнер, а не “первый попавшийся”.

Фаза 3 — “Closed loop”: переполнение ⇒ (fix) ⇒ (перепроверка) ⇒ (перевод-сжатие) (3–6 дней)

Ключевая инновация, которой нет в текущем плане: не пытаться всё решить шрифтом/spacing.

Алгоритм:

После translate/writeback прогнать validate_layout (у вас это есть как риск-детектор).

Для сегментов layout_*_overflow_risk:

Pass A: снять ограничения (ABBYY normalize)

Pass B: spacing/character spacing/font reduction (auto-fix)

Pass C (НОВЫЙ): если всё ещё риск высокий — второй LLM-проход “compress”:

“сократи перевод до N символов, сохрани плейсхолдеры ⟦…⟧ и терминологию”

N брать из реального approx_capacity_chars/extent

Ограничить “compress” только:

для заголовков/лейблов/текстбоксов/таблиц

и только при overflow>~1.2, чтобы не портить нормальный текст

Это даёт огромный выигрыш по “готовности”, потому что меньше случаев, когда Word вынужден ужимать шрифт до некрасивого.

Фаза 4 — Word COM как “финальный полировщик” (2–5 дней)

У вас уже есть режим mode=com и COM-постпроцессинг с параметрами:
com_textbox_min_font_pt, com_textbox_max_shrink_steps, com_expand_overflowing_shapes, com_textbox_max_height_growth 

config

.

Что сделать, чтобы это стало реально “production-grade”:

Включить com_expand_overflowing_shapes=true в ABBYY-пресете.

Добавить логику:

проверять Shape.TextFrame.Overflowing (истинный факт переполнения в Word)

пытаться увеличить высоту (в пределах max_height_growth) до уменьшения шрифта

уменьшение шрифта — только если расширение невозможно (упёрлись в коллизию/границы страницы)

Результат COM-прохода писать в QA-отчёт: какие shapes были “чинены”, на сколько изменён шрифт/размер.

Фаза 5 — Визуальная регрессия (чтобы не спорить “норм/не норм”) (1–3 дня)

Тебе нужно автоматизировать “человеческую проверку”.

У вас уже есть инфраструктура для рендеринга/preview сравнения страниц в merge_docx_sections.py (backend soffice|word, DPI, HTML) 

merge_docx_sections

 — это можно переиспользовать как визуальный QA:

рендер “до/после” в PNG

быстрый HTML-просмотр

автосигналы: рост числа страниц, сильный сдвиг элементов, “пустые” области (простая компьютерная метрика)

3) Оценка Claude-плана (что хорошо / что бы я поправил)

Сильное и верное:

правильно называет root-causes: runs rebuild 

claude_plan

, выключенные фиксы 

claude_plan

, ABBYY noAutofit/hRule="exact" 

claude_plan

, отсутствие ограничений контейнера в промпте 

claude_plan

.

пресеты + авто-детект — must have 

claude_plan

.

“SPACE_LIMIT” подсказка в LLM — реально рабочая идея 

claude_plan

.

Что бы я добавил/изменил:

Главный недостающий кусок: closed-loop “overflow ⇒ compress-retranslate”. Один шрифт/spacing не закрывает ABBYY-документы без деградации читаемости.

In-place writeback (предложение Claude) полезен, но я бы не ставил его №1:

сначала убрать ABBYY-жёсткости и сделать COM-детектор переполнения

иначе ты инвестируешь 4–5 дней, а обрезки останутся из-за noAutofit/exact

Детектор размеров контейнера надо сделать точным (см. Фаза 2B), иначе wp:extent будет иногда “чужой” и SPACE_LIMIT станет мусорным.

4) Конкретный “боевой” порядок внедрения (минимум риска)

Включить пресет “abbyy_full_com”: abbyy_profile=full, layout_auto_fix=true, passes=3, mode=com, com_expand_overflowing_shapes=true.

Сгенерировать QA-сводку: сколько было layout_*_overflow_risk до/после + сколько COM-исправлений.

Добавить SPACE_LIMIT в prompt (и только для textboxes/tables/frames).

Добавить compress-retranslate только для top-N проблемных сегментов (по overflow_ratio).

Уже потом — in-place writeback.