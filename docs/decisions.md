# Decisions

One dated entry per change that alters what a number or a verdict means. Append, never rewrite.

## 2026-10-10 — A figure the cited source does not carry fails the answer

- **What:** `rag/verify.py` checks, per cited sentence, that every figure the
  sentence states appears in the sources that sentence cites (thousand
  separators and decimal commas normalised; `[S1]` markers are not figures).
  A missing figure is listed in `unsupported_numbers` and makes the answer not
  `ok`. A sentence that negates while none of its cited sources negates is
  listed in `negation_mismatches` and blocks `strict_ok` only.
- **Why:** the lexical grounding score compares vocabulary, so "respond within
  999 hours [S1]" against a source saying 24 hours scored 0.83 and passed. A
  wrong amount, deadline or article number beside a valid citation is the
  failure a legal reader cannot see, and figures are the one thing a lexical
  check can judge with confidence. Negation is a heuristic (a source may negate
  elsewhere), hence strict only.
- **What changes in the numbers:** `ok` can now be false on an answer whose
  grounding score is above the threshold. The published eval tables were
  measured before this and are not re-run here; a re-run may show a lower
  `verify_ok` rate, which is the correction, not a regression.
- **What it does not settle:** scope ("applies to providers" when the source
  says deployers) and paraphrased contradictions still pass. Those need a
  claim-to-passage entailment check or a human reader.
- **Revisit if:** an answer legitimately derives a figure (a sum, a conversion)
  from its sources — it will be flagged; the fix is to cite the figures it was
  derived from, or to exempt derived figures explicitly.
