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

## Дополнительно зафиксированные паттерны part3

- protective-treatment таблицы:
  многострочные split-последовательности `Apply cadmium plate / Paint / Do not paint`,
  `Apply paint all over but not to`,
  `Apply only primer paint to`,
  `Paint area A`,
  `Before/After installation of bushes`
- surface-treatment и coating callouts:
  `Chromic acid anodise/anodize`,
  `Apply Alocrom 1200`,
  `Cadmium plate is optional on ...`,
  `No bare cadmium permitted`,
  `Apply zinc-nickel plate`,
  `Paint finish is optional in areas ...`
- figure/textbox выноски и техярлыки:
  `RUNOUT` -> `БИЕНИЕ`,
  `DEEP` -> `ГЛУБ.`,
  `THIS FACE ONLY` -> `ТОЛЬКО НА ЭТОЙ ПОВЕРХНОСТИ`,
  `INTERNALLY` -> `ВНУТРИ`,
  `SPOTFACE` -> `ПОДРЕЗКА ПЛОЩАДКИ`,
  `BORES` -> `ОТВЕРСТИЯ`
- спецификационные хвосты и split-reference фразы:
  `Type 1B / Type 2 / Class 1`,
  `from center`,
  `split housing / split line`,
  `thread / thread undercut / load bearing face`,
  `use three coat process ...`,
  `Apply top coat with Aerodur ...`

## Дополнительно зафиксированные паттерны part4

- таблицы утвержденных ремонтов:
  `Messier-Dowty Limited or Safran Landing Systems Repair No.` -> `Номер ремонта Messier-Dowty Limited или Safran Landing Systems`,
  `Applicable Part` -> `Применяемая деталь`
- figure-callout'ы по проушинам, отверстиям и поперечным сечениям:
  `KNUCKLE TOOLING LUG`,
  `TOOLING LUG`,
  `LOWER/UPPER DOOR LUGS`,
  `UPLOCK LUGS`,
  `LOWER CARDAN BORE`,
  `CROSS BORE FOR ...`,
  `CHANGE OVER VALVE HOLES AND LUGS`,
  `GREASE HOLES`,
  `BRAKE FLANGE`
- выноски по покрытиям и границам хромирования:
  `PAINT/ZINC-NICKEL DEPOSIT OVERLAP`,
  `FULL CHROME PLATING THICKNESS`,
  `CHROME RUN OUT BAND`,
  `DIA. AFTER GRINDING CHROME`,
  `DIA. AFTER CHROME PLATING`,
  `... CHROME TERMINATION`,
  `HPC SEAL ABUTMENT ... CHROME TERMINATION`
- служебная доводка cleanup:
  стартовое `and <ref>` больше не считается code-only хвостом и переводится,
  нормализуем `)-` -> `) -`,
  исправляем остатки вида `ФАСКАS`

## Дополнительно зафиксированные паттерны part6

- предупреждения и procedural callouts:
  `CAUTION: DO NOT USE A MECHANICAL MOP POLISHER TO GET THE SURFACE FINISH.`,
  `CAUTION: DO NOT MACHINE ALL OF THE FLANGE FACE.`,
  `CAUTION: REPAIR WILL NOT BE PERMITTED BEYOND THE LIMITS OF THIS REPAIR SCHEME.`,
  `CAUTION: FOR DAMAGE MORE THAN THE LIMITS OF THIS REPAIR SCHEME, WRITE TO SAFRAN LANDING SYSTEMS: REFER TO GUIDE-CS-001.`,
  `CAUTION: FOR DEVIATIONS OUTSIDE THE LIMITS OF THIS REPAIR SCHEME CONTACT SAFRAN LANDING SYSTEMS.`,
  `IF THE BASE METAL IS NOT DAMAGED`,
  `IF THE BASE METAL IS DAMAGED`
- figure-callout'ы по покрытиям и границам:
  `CHROMIUM PLATE DEPOSIT`,
  `CHROMIUM PLATING`,
  `MAX. CHROMIUM PLATE MUST TERMINATE IN THIS LENGTH.`,
  `CHROMIUM PLATE MUST TERMINATE WITHIN THIS LENGTH.`,
  `THE CHROMIUM PLATE MUST NOT EXTEND BEYOND THE DIMENSIONS SHOWN`,
  `TERMINATION TO M-DLPS1031-1/-5/-7`,
  `UNPLATED LENGTH`,
  `LENGTH OF NICKEL PLATE`,
  `LENGTH OF CHROMIUM`,
  `SERMETEL/SERMETAL COATING`,
  `FINISH PAINT`,
  tab-separated variant `Apply chromium plate to diameter A: refer to PCS-2110, type C and Figure 601.`
- IVD и материалы ремонтных втулок:
  `ALUMINIUM COAT OPTIONAL ON THESE SURFACES`,
  `APPLY COAT OF ALUMINIUM (IVD)`,
  `IVD COATING OPTIONAL`,
  `IVD COATING OPTIONAL IN END FACE`,
  `Aluminium Bronze`,
  `Bronze, UZ 19A6`,
  `Zinc Powder`
- геометрические выноски и размерные хвосты:
  `MINIMUM DIAMETER BEFORE CHROMIUM PLATE`,
  `MINIMUM WALL THICKNESS`,
  `MINIMUM LUG WIDTH`,
  `DEGREES`, `DEGREES REF.`, `MINUTES`, `RAD. REF.`, `REF.`,
  `FACE B/C/D/E/R`,
  `DIM. H`,
  `SMOOTH EDGE`,
  `EDGE SMOOTHED`,
  `EDGE SMOOTHED OUT`,
  `ENLARGED DETAIL`
- review cleanup:
  для проверки разрешены одиночные буквенные маркеры рисунков и product-name токены `Mastinox`, `Molykote`,
  кодоподобные обозначения вида `18-80A`/`09-510A` трактуем как допустимые reference/code сегменты, а не как непереведенный английский,
  uppercase English words длиной больше 4 символов больше не считаем code-like по умолчанию, чтобы review не пропускал callout-ы типа `CAUTION`

## Дополнительно зафиксированные паттерны part8

- процедуры ремонта корпуса стойки:
  `Blank Bush`,
  `Oversize Bushes For Diameter B`,
  `With the measured dimension E, select the applicable oversize bush from Table 1.`,
  `Check line ream the repair bushes ...`,
  `Apply flash chromium plate ...`,
  `Select the applicable oversize lubrication adaptor ...`,
  `Install the selected lubrication adaptor(s) ...`
- figure-callout'ы и размерные подписи:
  `OVERSIZE BUSH 450237810`,
  `BLEND SMOOTHLY TO ADJACENT SURFACES`,
  `MINIMUM WALL THICKNESS`,
  `SECTION Z-Z`,
  `DETAIL Y`,
  `LUG WIDTH D`,
  `POINT`,
  `4 HOLES`,
  `REMOVE EDGES`,
  `NO PAINT`
- embedded figure cleanup:
  если английские выноски находятся не в paragraph/textbox-слое, а внутри `word/media/*`,
  допускается патчить сам media asset и сохранять отдельный manual-fix report,
  после такого патча обязателен повторный render + review + OCR/визуальный контроль проблемной страницы

## Дополнительно зафиксированные паттерны part9

- ремонты корпуса стойки `11-25`...`11-33`:
  точные формулировки для процедур oversize lubrication adaptor,
  oversize lower bearing subassembly,
  oversize spherical bearing assembly,
  а также для записей `Record the repair number...`, `Apply cadmium plate all over but not to the areas shown`,
  `Apply primer and top coat paint ...`, `Apply top coat paint ...`
- таблицы oversize-adaptor и lower-bearing:
  `Oversize Step mm (in)`,
  `Oversize / Lubrication Adaptor Number`,
  `Before chromium / plate mm (in)`,
  `After chromium / plate mm (in)`,
  `Production`,
  порядковые ступени `1st`...`12th`,
  `Inner Diameter A Before Plating mm (in)`,
  `Oversize Bearing Housing Outer Diameter Ref. mm (in)`,
  `Spherical Bearing Assembly Number`
- материалы и спецификации:
  `Loctite Grade 270` -> `Loctite, марка 270`,
  `Zinc loaded Molykote 111` -> `Molykote 111 с цинковым наполнителем`,
  `UHT Steel ... with UTS ...` -> перевод через `временное сопротивление разрыву`,
  `Steel ... heat treated to S154` -> `термообработанная до состояния S154`
- figure/callout cleanup:
  `REPAIR LUBRICATION ADAPTOR`,
  `CHROME FADE OUT`,
  `TO INTERSECTION`,
  `CADMIUM PLATE OPTIONAL AND NO PAINT.`,
  `CHROMIUM PLATE TERMINATION ...`,
  `WITH INNER LINER`,
  `CORRECT OVERSIZE ...`,
  `OVERSIZE O RING / BACKING RING / TRANSFER DOWEL`
