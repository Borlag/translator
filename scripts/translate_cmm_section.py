from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.manual_translate_part1 import (
    run_translation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Translate a CMM section DOCX using the shared rulebase.")
    parser.add_argument("--source", required=True, help="Path to the source DOCX section")
    parser.add_argument("--output", required=True, help="Path to the translated DOCX section")
    parser.add_argument("--report", required=True, help="Path to the JSON report")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    report = Path(args.report).resolve()
    report_payload = run_translation(source, output, report)
    print(json.dumps(report_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
