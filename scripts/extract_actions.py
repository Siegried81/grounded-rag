"""Extract the actions of one meeting transcript and print them as JSON.

Usage: python scripts/extract_actions.py TRANSCRIPT.txt --meeting-date 2026-03-02
       [--directory eval/actions/directory.json]

A thin command line over `rag.actions.extract_actions`: one LLM call, the Python
checks, and every action printed with its owner id and review flags. The exit
code is 1 when the extraction failed, so a script calling it cannot mistake a
failure for a meeting with no actions.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from rag.actions import extract_actions, load_directory  # noqa: E402
from rag.logging_utils import never_crash_on_console_encoding  # noqa: E402


def main() -> int:
    """Parse arguments, extract, print JSON; return the process exit code."""
    never_crash_on_console_encoding()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("transcript")
    parser.add_argument("--meeting-date", required=True, type=date.fromisoformat)
    parser.add_argument("--directory",
                        default=str(config.ROOT / "eval" / "actions" / "directory.json"))
    args = parser.parse_args()
    result = extract_actions(Path(args.transcript).read_text(encoding="utf-8"),
                             args.meeting_date, load_directory(Path(args.directory)))
    print(json.dumps({
        "provider": result.provider,
        "error": result.error,
        "actions": [{**asdict(a), "needs_review": a.needs_review} for a in result.actions],
    }, ensure_ascii=False, indent=2))
    return 1 if result.error else 0


if __name__ == "__main__":
    sys.exit(main())
