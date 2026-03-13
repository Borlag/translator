from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.docxru.config import load_config
from src.docxru.docx_reader import collect_segments
from src.docxru.llm import build_glossary_matchers, build_llm_client
from src.docxru.tagging import paragraph_to_tagged, tagged_to_runs
from src.docxru.token_shield import PatternRule, PatternSet, shield, unshield


MANUAL_SOURCE_GLOSSARY = """
| Material Ref. Item | материал, поз. |
| anti-corrosion compound | антикоррозионный состав |
| TESTING AND FAULT ISOLATION | ПРОВЕРКА И ПОИСК НЕИСПРАВНОСТЕЙ |
| Testing and Fault Isolation | Проверка и поиск неисправностей |
| Press Pad | прижимная пластина |
| Drift | выколотка |
| Torque Adapter | моментный адаптер |
| Jacking Dome | Поддомкратный купол |
| jacking dome | поддомкратный купол |
| anti-tamper sealant | контровочный герметик |
| Main Fitting | корпус стойки |
| main fitting | корпус стойки |
| subassembly | подсборка |
| sub-assembly | подсборка |
| Related Parts | соответствующие детали |
| related parts | соответствующие детали |
| Upper bearing housing | корпус верхнего подшипника |
| upper bearing housing | корпус верхнего подшипника |
| retainer ring | стопорное кольцо |
| Retainer ring | Стопорное кольцо |
| locking pins | фиксирующие штифты |
| Locking pins | Фиксирующие штифты |
| locking plates | стопорные пластины |
| Locking plates | Стопорные пластины |
| tab washers | стопорные шайбы |
| Tab washers | Стопорные шайбы |
| two piece stop | двухсекционный упор |
| two piece stop with inserts | двухсекционный упор со вставками |
| recoil orifice plate | пластина обратного дросселя |
| Recoil orifice plate | Пластина обратного дросселя |
| sliding tube subassembly | подсборка скользящей трубки |
| Sliding Tube Subassembly | Подсборка скользящей трубки |
| sliding piston subassembly | подсборка скользящего поршня |
| Upper Diaphragm Tube Subassembly | Подсборка верхней диафрагменной трубки |
| upper diaphragm tube subassembly | подсборка верхней диафрагменной трубки |
| Shock Absorber Subassembly | Подсборка амортизатора |
| charging valve | заправочный клапан |
| inflation valve subassembly | подсборка клапана накачки |
| Crowfoot Wrench | ключ Crowfoot |
| stop rings | стопорные кольца |
| split pins | шплинты |
| Sleeve | Втулка |
| SLEEVE | ВТУЛКА |
| Repair Sleeve | Ремонтная втулка |
| Repair Sleeves | Ремонтные втулки |
"""


MANUAL_EXACT_MAP: dict[str, str] = {
    "Refer to M-DLPS1005-1. Lubricate these parts with hydraulic fluid, Material Ref. Item 02-501:":
        "См. M-DLPS1005-1. Смажьте эти детали гидравлической жидкостью, материал, поз. 02-501:",
    "NOTE: You can lubricate the seals with grease, Mobil 28, Material Ref. Item 04-526: refer to M-DLPS1011-1.":
        "ПРИМЕЧАНИЕ: Уплотнения можно смазать смазкой Mobil 28, материал, поз. 04-526: см. M-DLPS1011-1.",
    "Refer to M-DLPS709-14. Apply anti-corrosion compound, Material Ref. Item TBA, to these parts:":
        "См. M-DLPS709-14. Нанесите антикоррозионный состав, материал, поз. TBA, на следующие детали:",
    "Refer to M-DLPS709-14. Apply anti-corrosion compound, Material Ref. Item TBA, to these areas:":
        "См. M-DLPS709-14. Нанесите антикоррозионный состав, материал, поз. TBA, на следующие участки:",
    "Refer to M-DLPS709-14. Apply anti-corrosion compound, Material Ref. Item TBA, to:":
        "См. M-DLPS709-14. Нанесите антикоррозионный состав, материал, поз. TBA, на:",
    "Refer to M-DLPS709-11. Apply anti-corrosion compound, Material Ref. Item TBA, to:":
        "См. M-DLPS709-11. Нанесите антикоррозионный состав, материал, поз. TBA, на:",
    "Do the piston leakage test: refer to TESTING AND FAULT ISOLATION.":
        "Выполните проверку поршня на герметичность: см. ПРОВЕРКА И ПОИСК НЕИСПРАВНОСТЕЙ.",
    "Do the piston leakage test. Refer to TESTING AND FAULT ISOLATION.":
        "Выполните проверку поршня на герметичность. См. ПРОВЕРКА И ПОИСК НЕИСПРАВНОСТЕЙ.",
    "Repeat the above step.":
        "Повторите предыдущий шаг.",
    "Install the locking plates (15-80)":
        "Установите стопорные пластины (15-80).",
    "NOTE: It is necessary to drill holes in new locking plates (15-80).":
        "ПРИМЕЧАНИЕ: В новых стопорных пластинах (15-80) необходимо просверлить отверстия.",
    "Install the locking plates (15-80) in the upper bearing housing (15-40A) on the two piece stop subassembly (15-110).":
        "Установите стопорные пластины (15-80) в корпус верхнего подшипника (15-40A) на подсборку двухсекционного упора (15-110).",
    "Install the screws (15-90) and the tab washers (15-100) through the locking plates (15-80) into the two piece stop with inserts (15-130).":
        "Установите винты (15-90) и стопорные шайбы (15-100) через стопорные пластины (15-80) в двухсекционный упор со вставками (15-130).",
    "Slide the retainer ring (15-60) and the recoil orifice plate (15-70) inside the upper bearing housing (15-40A) to align the holes in the upper bearing housing (15-40A) with the holes in the retainer ring (15-60).":
        "Сдвиньте стопорное кольцо (15-60) и пластину обратного дросселя (15-70) внутрь корпуса верхнего подшипника (15-40A), чтобы совместить отверстия в корпусе верхнего подшипника (15-40A) с отверстиями в стопорном кольце (15-60).",
    "Attach the locking pins (15-50) through the retainer ring (15-60) to the upper bearing housing (15-40A).":
        "Установите фиксирующие штифты (15-50) через стопорное кольцо (15-60) в корпус верхнего подшипника (15-40A).",
    "Install the upper bearing housing (14-70), the retainer ring (15-60) and the recoil orifice plate (15-70) to the sliding piston subassembly (17-240B) or (17-240C) or (17-240F) or (17-240G).":
        "Установите корпус верхнего подшипника (14-70), стопорное кольцо (15-60) и пластину обратного дросселя (15-70) на подсборку скользящего поршня (17-240B), (17-240C), (17-240F) или (17-240G).",
    "Install the bearings (15-20) and (15-30) over the upper bearing housing (15-40A).":
        "Установите подшипники (15-20) и (15-30) на корпус верхнего подшипника (15-40A).",
    "Install the Sliding Tube Subassembly (17-240) and its Related Parts":
        "Установите подсборку скользящей трубки (17-240) и относящиеся к ней детали.",
    "Install the stop rings (14-70).":
        "Установите стопорные кольца (14-70).",
    "Temporarily install the nuts (14-60).":
        "Временно установите гайки (14-60).",
    "Temporarily install the bolts (14-50), the washers (14-40), the nuts (14-30) and the split pins (14-20). Open the split pins (14-20) sufficiently to keep in place.":
        "Временно установите болты (14-50), шайбы (14-40), гайки (14-30) и шплинты (14-20). Разведите шплинты (14-20) настолько, чтобы они удерживались на месте.",
    "The Total Required column (TTL REQ.) shows the total necessary each time the part number is shown in the Detailed Parts List.":
        "Столбец «Общее требуемое количество» (TTL REQ.) показывает общее количество, необходимое каждый раз, когда номер детали указан в Подробном перечне деталей.",
    "The Effectivity Code (EFF. CODE) agrees with that of the next higher assembly. The effectivity code also shows if subassemblies and details are applicable to their next higher assembly or subassembly. When an item is applicable to all units the Effectivity Code column will be empty. The effectivity code usage is specific to the IPL figure to which it applies.":
        "Код применяемости (EFF. CODE) совпадает с кодом следующей вышестоящей сборки. Код применяемости также показывает, относятся ли подсборки и детали к своей следующей вышестоящей сборке или подсборке. Если элемент применим ко всем экземплярам, столбец кода применяемости остается пустым. Использование кода применяемости относится только к тому рисунку IPL, к которому он применяется.",
    "The quantity in the Units per Assembly column is the quantity necessary for the next higher assembly. AR in the Units per Assembly column shows that the quantity of parts to be used is as required. RF in the Units per Assembly column shows that the part is for reference only.":
        "Количество в столбце Units per Assembly показывает количество, необходимое для следующей вышестоящей сборки. Обозначение AR в столбце Units per Assembly означает, что количество деталей определяется по необходимости. Обозначение RF в столбце Units per Assembly означает, что деталь приведена только для справки.",
    "The Part Numbers that are shown (NP) in the Detailed Parts List are non-procurable items. Unless the part has been superseded the next higher assembly must be installed.":
        "Номера деталей, помеченные (NP) в Подробном перечне деталей, обозначают непоставляемые позиции. Если деталь не заменена новой, должна устанавливаться следующая вышестоящая сборка.",
    "1,50mm (0.059in)": "1,50mm (0.059in)",
    "129,75mm (5.1083in) MIN": "129,75mm (5.1083in) MIN",
    "A321A5906-1": "A321A5906-1",
    "Y": "Y",
    "C": "C",
    "SLEEVE (5-210)": "ВТУЛКА (5-210)",
}

LOCATION_TOUCHUPS: dict[str, str] = {
    "body/p2757": "Ремонт № 15-2",
    "body/p9176":
        "Вставьте стопорное кольцо (15-60) и пластину обратного дросселя (15-70) внутрь корпуса верхнего подшипника (15-40A), чтобы совместить отверстия в корпусе верхнего подшипника (15-40A) с отверстиями в стопорном кольце (15-60).",
    "body/p9177":
        "Установите фиксирующие штифты (15-50) через стопорное кольцо (15-60) в корпус верхнего подшипника (15-40A).",
    "body/p9178":
        "Установите корпус верхнего подшипника (14-70), стопорное кольцо (15-60) и пластину обратного дросселя (15-70) на подсборку скользящего поршня (17-240B), (17-240C), (17-240F) или (17-240G).",
    "body/p9194":
        "Выполните проверку поршня на герметичность. См. раздел «ПРОВЕРКА И ПОИСК НЕИСПРАВНОСТЕЙ».",
    "body/p11795":
        "Количество в столбце «Количество на сборку» показывает количество, необходимое для следующей вышестоящей сборки. Обозначение AR в столбце «Количество на сборку» означает, что количество деталей определяется по необходимости. Обозначение RF в столбце «Количество на сборку» означает, что деталь приведена только для справки.",
    "body/p8351":
        "Затяните гайку заправочного клапана (17-20) моментом от 5,7 до 7,9 N m (от 4.2 до 5.82 lbf ft).",
    "body/p8709": "92,85-93,00mm (3.655-3.661in)",
    "body/p8711": "64,650-64,775mm (2.5453-2.5502in)",
    "body/p8972":
        "Установите болт (15-350), шайбу (15-340) и гайку (15-330). Затяните гайку (15-330) с моментом от 25 до 29 N m (18.5 и 21.5 lbf ft).",
    "body/p7409": "MESSIER-DOWTY Gloucester",
    "body/textbox211/p0": "ДИАМ. B",
    "body/textbox360/p0": "1,00-3,00mm",
}


EXTRA_SHIELD_RULES = [
    PatternRule(name="ABBR", pattern=r"\bTBA\b"),
    PatternRule(name="ABBR", pattern=r"\bTTL REQ\.\b"),
    PatternRule(name="ABBR", pattern=r"\bEFF\. CODE\b"),
    PatternRule(name="ABBR", pattern=r"\bNP\b"),
    PatternRule(name="ABBR", pattern=r"\bAR\b"),
    PatternRule(name="ABBR", pattern=r"\bRF\b"),
    PatternRule(name="UNIT", pattern=r"\bN m\b"),
    PatternRule(name="UNIT", pattern=r"\blbf ft\b"),
    PatternRule(name="UNIT", pattern=r"\blbf in\b"),
    PatternRule(name="UNIT", pattern=r"(?<=\d)\s*mm\b"),
    PatternRule(name="UNIT", pattern=r"(?<=\d)\s*in\b"),
]


def _target_text(seg) -> str:
    return str(seg.source_plain or "").strip()


def _is_broken(seg) -> bool:
    text = _target_text(seg)
    return bool(text) and "?" in text


def _needs_copy_as_is(source: str) -> bool:
    stripped = source.strip()
    if not stripped:
        return False
    if stripped in MANUAL_EXACT_MAP:
        return False
    if len(stripped) == 1 and stripped.isupper():
        return True
    if all(ch.isdigit() or ch in " .,()/-ABCDEFGHIJKLMNOPQRSTUVWXYZ" for ch in stripped) and any(
        ch.isdigit() for ch in stripped
    ):
        return True
    return False


def _build_single_span_tagged(text: str, source_tagged: str, spans) -> str:
    if not spans:
        return text
    first = spans[0]
    flag_part = "|" + "|".join(first.flags) if first.flags else ""
    return f"⟦S_{first.span_id}{flag_part}⟧{text}⟦/S_{first.span_id}⟧"


def _apply_location_touchup(seg, source_seg, *, logger: logging.Logger) -> bool:
    replacement = LOCATION_TOUCHUPS.get(seg.location)
    if not replacement:
        return False
    source_tagged, spans, inline_map = paragraph_to_tagged(source_seg.paragraph_ref)
    tagged_to_runs(
        seg.paragraph_ref,
        _build_single_span_tagged(replacement, source_tagged, spans),
        spans,
        inline_run_map=inline_map,
    )
    logger.info("touchup %s", seg.location)
    return True


def _apply_manual_source_glossary(text: str, replacements) -> str:
    out = text
    for _, target, pattern in replacements:
        out = pattern.sub(target, out)
    return out


def _postprocess_translated_text(text: str) -> str:
    out = text
    out = out.replace("–", "-").replace("—", "-").replace("−", "-")
    out = out.replace("Н·м", "N m").replace("Н м", "N m")
    out = out.replace("lbf фут", "lbf ft")
    out = out.replace("lbf футов", "lbf ft")
    out = out.replace("lbf-фут", "lbf ft")
    out = out.replace("lbf дюйм", "lbf in")
    out = out.replace("lbf-дюйм", "lbf in")
    out = out.replace("в эти части", "на следующие детали")
    out = out.replace("к этим частям", "на следующие детали")
    out = out.replace("в эти области", "на следующие участки")
    out = out.replace("к этим областям", "на следующие участки")
    out = out.replace("Применять ", "Нанесите ")
    out = out.replace("применять ", "нанесите ")
    out = out.replace("на место соединения", "на стык")
    out = out.replace("Подсборка", "подсборка")
    return out


def _cleanup_target_text(text: str) -> str:
    out = text
    out = re.sub(r"\bRepair No\.\s*", "Ремонт № ", out)
    out = re.sub(r"\bLoctite Grade\b", "Loctite марки", out)
    out = re.sub(r"\bType ([0-9A-Z]+), Class (\d+)\b", r"тип \1, класс \2", out)
    out = re.sub(r"\bNm\b", "N m", out)
    out = re.sub(r"\bDIA\.", "ДИАМ.", out)
    out = re.sub(r"\bDIM\.", "РАЗМ.", out)
    out = re.sub(r"\bMIN\.?\b", "МИН.", out)
    out = re.sub(r"\bMAX\.?\b", "МАКС.", out)

    letters = re.findall(r"[A-Za-z]+", out)
    meaningful = [token for token in letters if token.lower() not in {"mm", "in", "lbf", "ft", "pcs", "mil", "ampep"}]
    if not meaningful or all(token.lower() in {"to", "min", "max", "dia", "dim"} for token in meaningful):
        out = re.sub(r"(?<=\d)\s+to\s+(?=[\d(])", "-", out)
    return out


def _apply_target_cleanup(seg, *, logger: logging.Logger) -> bool:
    current = str(seg.paragraph_ref.text or "")
    updated = _cleanup_target_text(current)
    if updated == current:
        return False
    current_tagged, spans, inline_map = paragraph_to_tagged(seg.paragraph_ref)
    tagged_to_runs(
        seg.paragraph_ref,
        _build_single_span_tagged(updated, current_tagged, spans),
        spans,
        inline_run_map=inline_map,
    )
    logger.info("cleanup %s", seg.location)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair broken section 2 segments with Google fallback.")
    parser.add_argument("--input", required=True, help="Path to source English DOCX.")
    parser.add_argument("--output", required=True, help="Path to translated Russian DOCX.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--report", required=True, help="JSON report path.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit of repaired segments.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("google_fix_part2_questions")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler(sys.stdout))

    cfg = load_config(str(Path(args.config).resolve()))
    if not output_path.exists():
        shutil.copyfile(input_path, output_path)

    source_doc = collect_segments(
        Document(str(input_path)),
        include_headers=cfg.include_headers,
        include_footers=cfg.include_footers,
    )
    output_document = Document(str(output_path))
    output_segments = collect_segments(
        output_document,
        include_headers=cfg.include_headers,
        include_footers=cfg.include_footers,
    )

    source_by_location = {seg.location: seg for seg in source_doc if seg.location}
    broken_segments = [seg for seg in output_segments if _is_broken(seg) and seg.location in source_by_location]
    if args.limit > 0:
        broken_segments = broken_segments[: int(args.limit)]

    glossary_path = Path(cfg.llm.glossary_path)
    if not glossary_path.is_absolute():
        glossary_path = Path(args.config).resolve().parent / glossary_path
    glossary_text = glossary_path.read_text(encoding="utf-8")
    full_glossary_text = glossary_text + "\n" + MANUAL_SOURCE_GLOSSARY
    manual_replacements = build_glossary_matchers(MANUAL_SOURCE_GLOSSARY)
    llm_client = build_llm_client(
        provider="google",
        model="",
        temperature=0.0,
        timeout_s=max(60.0, float(cfg.llm.timeout_s)),
        max_output_tokens=max(512, int(cfg.llm.max_output_tokens)),
        glossary_text=full_glossary_text,
    )
    pattern_set = PatternSet(rules=[*cfg.pattern_set.rules, *EXTRA_SHIELD_RULES])

    applied: list[dict[str, str]] = []
    failed: list[dict[str, str]] = []

    logger.info("Broken segments selected: %d", len(broken_segments))

    for index, out_seg in enumerate(broken_segments, start=1):
        source_seg = source_by_location[out_seg.location]
        source_plain = str(source_seg.source_plain or "").strip()
        target_before = _target_text(out_seg)
        if not source_plain:
            failed.append({"location": out_seg.location, "reason": "empty_source"})
            continue

        try:
            source_tagged, spans, inline_map = paragraph_to_tagged(source_seg.paragraph_ref)
        except Exception as exc:
            failed.append({"location": out_seg.location, "reason": f"tagging_failed: {exc}"})
            continue

        try:
            if source_plain in MANUAL_EXACT_MAP:
                translated_tagged = _build_single_span_tagged(MANUAL_EXACT_MAP[source_plain], source_tagged, spans)
            elif _needs_copy_as_is(source_plain):
                translated_tagged = _build_single_span_tagged(source_plain, source_tagged, spans)
            else:
                prepared = _apply_manual_source_glossary(source_tagged, manual_replacements)
                shielded, token_map = shield(prepared, pattern_set)
                translated_tagged = unshield(
                    llm_client.translate(
                        shielded,
                        {
                            "task": "translate",
                            "structured_layout": bool(out_seg.location.startswith("body/textbox")),
                            "preserve_line_breaks": True,
                        },
                    ),
                    token_map,
                )
                translated_tagged = _postprocess_translated_text(translated_tagged)

            tagged_to_runs(
                out_seg.paragraph_ref,
                translated_tagged,
                spans,
                inline_run_map=inline_map,
            )
            after_text = out_seg.paragraph_ref.text or ""
            applied.append(
                {
                    "location": out_seg.location,
                    "source": source_plain,
                    "before": target_before,
                    "after": after_text,
                }
            )
            logger.info("[%d/%d] fixed %s", index, len(broken_segments), out_seg.location)
        except Exception as exc:
            failed.append({"location": out_seg.location, "reason": str(exc), "source": source_plain})

    for out_seg in output_segments:
        source_seg = source_by_location.get(out_seg.location)
        if source_seg is None:
            continue
        try:
            _apply_location_touchup(out_seg, source_seg, logger=logger)
        except Exception as exc:
            failed.append({"location": out_seg.location, "reason": f"touchup_failed: {exc}"})

    for out_seg in output_segments:
        try:
            _apply_target_cleanup(out_seg, logger=logger)
        except Exception as exc:
            failed.append({"location": out_seg.location, "reason": f"cleanup_failed: {exc}"})

    output_document.save(str(output_path))

    remaining = sum(1 for seg in output_segments if "?" in str(seg.paragraph_ref.text or ""))

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "selected_segments": len(broken_segments),
        "applied_segments": len(applied),
        "failed_segments": len(failed),
        "remaining_question_segments": remaining,
        "applied": applied,
        "failed": failed,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Report written: %s", report_path)
    logger.info("Remaining question segments: %d", remaining)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
