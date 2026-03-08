# CMM Translation Decisions

Этот файл фиксирует решения, которые должны повторно использоваться для всех частей одного и того же документа.

## Контекст

- glossary: `for_test/new_formating/glossary.md`
- general prompt: `for_test/new_formating/general_prompt.md`
- shared translation rulebase: `scripts/manual_translate_part1.py`
- generic runner: `scripts/translate_cmm_section.py`
- review runner: `scripts/review_cmm_translation.py`

## Терминология

- `Main Landing Gear Leg` -> `Стойка основного шасси`
- `Main Fitting` -> `Корпус стойки`
- `Torque Link` -> `Шлиц-шарнир`
- `Key Diagram` -> `Схема расположения`
- `Protective Treatment` -> `Защитная обработка`
- `Repair Procedure Conditions` -> `Условия выполнения процедуры ремонта`
- `Illustrated Parts List` -> `Иллюстрированный перечень деталей`

## Нормализация оформления

- даты формата `Mon DD/YYYY` переводим в `DD.MM.YYYY`
- диапазоны `to` между числами переводим в дефис
- связку `and` между числами переводим в перечисление через запятую
- `Page` -> `Стр.`
- `No.` -> `№`
- в многострочных ячейках таблиц переводим служебные хвосты и разрывы построчно:
  `M-D Spec` -> `Спецификация M-D`,
  отдельное `and` -> `и`,
  `Type 1 or 17-4PH to` -> `тип 1 или 17-4PH по`,
  `Use with ... and ...` -> `Использовать с ... и ...`
- для таблиц после layout auto-fix обязательно поднимаем все run'ы минимум до `9 pt`
  и уплотняем абзацы в ячейках; шрифт ниже `9 pt` не допускается

## Что сохраняем без перевода

- коды деталей и reference IDs
- официальные названия компаний, если они выступают собственным именем
- URL и серийные/каталожные идентификаторы

## Правило пополнения базы

Если при переводе новой части найден повторяемый английский фрагмент, который реально относится к тексту документа, а не к коду детали, правило нужно добавлять в shared rulebase и затем повторно прогонять текущую часть.

## Дополнительно зафиксированные паттерны part2

- специальные инструменты:
  `Lifting Bar Assembly`, `Pintle Location Assembly`, `Spherical Bearing Locator`,
  `Assembly/Extraction Tool`, `Extractor Pad and Drawbolt`,
  `Torque Reaction Adapter`, `Pin Spanner`
- disassembly-паттерны:
  `Release the tab washer`,
  `slotted nut`,
  `its related parts`,
  `its attached parts`,
  `forward pintle bush`,
  `bonding cable`,
  `identification washer(s)`
- материалы и NDT-таблицы:
  многострочные `PCS-3100 / M-DLNDT3 / Parts 1 and 2`,
  `and PCS-3002`,
  `Stainless Steel, Z15CN17-03 / Type 1 / 17-4PH / AMS 5604/5643`
