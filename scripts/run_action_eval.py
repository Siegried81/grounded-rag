"""Measure the action extraction (rag/actions.py) on the hand-labelled meetings in eval/actions/.

Each meeting is sent through `extract_actions` once, exactly as a user would
run it, and the extracted actions are compared with the labelled ones.

How a predicted action is paired with a labelled one: by its source sentence.
Both sentences go through `normalize_text`, and a pair counts when the word
overlap (Jaccard) is at least `MATCH_MIN`; each labelled action is used once,
best pairs first. The labelled set keeps one action per sentence, so the source
sentence identifies the action without judging how the title was worded.

What is reported, per run:
- precision and recall on actions (with 95% Wilson intervals: the set is small,
  and the interval says how small);
- owner accuracy and due-date accuracy on the paired actions. A labelled owner
  of null means "nobody can be assigned" (an unknown or ambiguous name), so the
  right output there is no owner id;
- wrong owner assigned: paired actions given an owner id that is not the
  labelled one. This is the costly error (a task sent to the wrong person), so
  it gets its own line;
- errors caught by review: of the paired actions with a wrong owner or date,
  how many carry at least one flag, i.e. would have reached a person before
  being sent;
- extra actions caught by review: the same for predicted actions that pair
  with no labelled one (invented, cancelled, already done);
- meetings that failed (LLM error or invalid JSON), which count as zero actions
  found, not as zero actions expected.

`--out` keeps every extracted action with its flags, for audit.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from rag.actions import ReviewedAction, extract_actions, load_directory  # noqa: E402
from rag.answer_metrics import normalize_text  # noqa: E402
from rag.logging_utils import never_crash_on_console_encoding  # noqa: E402
from rag.stats import format_rate  # noqa: E402

EVAL_DIR = config.ROOT / "eval" / "actions"
DEFAULT_OUT = config.ROOT / "logs" / "action_eval_results.jsonl"
MATCH_MIN = 0.5


def _jaccard(a: str, b: str) -> float:
    """Word-set overlap of two sentences after normalisation."""
    x, y = set(normalize_text(a).split()), set(normalize_text(b).split())
    return len(x & y) / len(x | y) if x | y else 0.0


def pair_actions(predicted: list[ReviewedAction], gold: list[dict]) -> list[tuple[int, int]]:
    """(predicted index, gold index) pairs, best overlap first, each side used once."""
    scored = sorted(
        ((_jaccard(p.source_sentence, g["source_sentence"]), i, j)
         for i, p in enumerate(predicted) for j, g in enumerate(gold)),
        reverse=True,
    )
    used_p, used_g, pairs = set(), set(), []
    for score, i, j in scored:
        if score < MATCH_MIN:
            break
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            pairs.append((i, j))
    return sorted(pairs)


def score_meeting(predicted: list[ReviewedAction], gold: list[dict]) -> Counter:
    """Counts for one meeting; `summarize` turns the totals into rates."""
    c = Counter(predicted=len(predicted), gold=len(gold))
    pairs = pair_actions(predicted, gold)
    paired = {i for i, _ in pairs}
    extra = [p for i, p in enumerate(predicted) if i not in paired]
    c["extra"] = len(extra)
    c["extra_flagged"] = sum(p.needs_review for p in extra)
    for i, j in pairs:
        p, g = predicted[i], gold[j]
        owner_ok = p.owner_id == g["owner_id"]
        date_ok = p.due_date == g["due_date"]
        c["matched"] += 1
        c["owner_ok"] += owner_ok
        c["date_ok"] += date_ok
        c["wrong_owner_assigned"] += p.owner_id is not None and not owner_ok
        if not (owner_ok and date_ok):
            c["errors"] += 1
            c["errors_flagged"] += p.needs_review
    c["flagged"] = sum(p.needs_review for p in predicted)
    return c


def summarize(totals: Counter) -> list[tuple[str, str]]:
    """Rows of the printed summary, every rate shown as k/n with its interval."""
    return [
        ("meetings failed", str(totals["failed"])),
        ("precision (actions)", format_rate(totals["matched"], totals["predicted"])),
        ("recall (actions)", format_rate(totals["matched"], totals["gold"])),
        ("owner accuracy (paired)", format_rate(totals["owner_ok"], totals["matched"])),
        ("due date accuracy (paired)", format_rate(totals["date_ok"], totals["matched"])),
        ("wrong owner assigned", format_rate(totals["wrong_owner_assigned"], totals["matched"])),
        ("errors caught by review", format_rate(totals["errors_flagged"], totals["errors"])),
        ("extra actions caught", format_rate(totals["extra_flagged"], totals["extra"])),
        ("actions flagged for review", format_rate(totals["flagged"], totals["predicted"])),
    ]


def main() -> None:
    """Run every labelled meeting through the extractor and print the summary."""
    never_crash_on_console_encoding()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-dir", default=str(EVAL_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    directory = load_directory(eval_dir / "directory.json")
    items = [json.loads(line) for line in
             (eval_dir / "actions_eval.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip()]
    totals, providers = Counter(), Counter()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in items:
            transcript = (eval_dir / "transcripts" / item["transcript"]).read_text(encoding="utf-8")
            result = extract_actions(transcript, date.fromisoformat(item["meeting_date"]),
                                     directory)
            providers[result.provider] += 1
            counts = score_meeting(result.actions, item["actions"])
            counts["failed"] = int(result.error is not None)
            totals.update(counts)
            f.write(json.dumps({"id": item["id"], "provider": result.provider,
                                "error": result.error,
                                "actions": [asdict(a) for a in result.actions],
                                "counts": dict(counts)}, ensure_ascii=False) + "\n")
            status = f"ERROR {result.error[:60]}" if result.error else (
                f"{counts['matched']}/{counts['gold']} found, {counts['predicted']} extracted, "
                f"{counts['flagged']} flagged")
            print(f"{item['id']:<4} {status}")
    print("-" * 70)
    print(f"answered by: {dict(providers)}")
    for label, value in summarize(totals):
        print(f"{label:<28} {value}")
    print(f"results: {out}")


if __name__ == "__main__":
    main()
