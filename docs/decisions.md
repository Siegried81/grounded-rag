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

## 2026-10-14 — Figure check: whole numbers keep their zeros, scale words and no-break spaces understood

- **What:** `verify._numbers` stripped trailing zeros from every figure, not just
  from decimals, so "35 000 000" read as "35", "1,000" as "1" and "240 hours" as
  "24 hours". Only a decimal part is trimmed now ("2.50" = "2.5"). The thousands
  separator set adds U+00A0 and U+202F, the no-break spaces EUR-Lex and gpt-oss
  use. A figure followed by a scale word (million, billion, milliard, miljoen…)
  is supported by either its expanded value ("15 million" = Article 99's
  "15 000 000") or its written value ("$416,161 million" = a 10-K table in
  millions).
- **Why:** the zero stripping let a wrong figure pass (an answer saying 240 hours
  against a source saying 24 was "supported"), and the missing separators made a
  correct figure fail (aa-03's spurious "0").
- **What changes in the numbers:** re-verifying the 40 stored answers that carry
  sources (`logs/answer_eval_results.jsonl`, `logs/redteam_results.jsonl`, full
  source files) changes one verdict: aa-03 goes back to `ok`, its spurious "0"
  gone. No published table is re-run here. Tests 418 → 429 passing.
- **Revisit if:** a source states its unit away from the figure ("in millions"
  in a table header) and the answer gives the expanded value: that pair is still
  reported as unsupported.

## 2026-10-14 — Answer and red-team evaluations re-run live

- **What:** `run_eval.py` (k = 3 and 5, both corpora), `run_phrasing_eval.py`,
  `run_answer_eval.py` and `run_redteam.py` re-run end to end with
  `openai/gpt-oss-120b` on Groq and `nomic-embed-text` on Ollama, with every
  scorer change since the previous run in force. The answer and red-team runs
  were made with `LLM_CACHE=0`: a first pass with the cache on replayed most of
  the 6 October answers (11 new cache entries for 61 calls) and is discarded. The README and deep-dive tables
  now report this run; the projections they used to carry are kept as history.
- **Why:** three scorer changes (key-fact matcher, figure and negation checks,
  refusal phrases) had landed after the published runs, so the tables described
  code that no longer existed.
- **What changes in the numbers:** retrieval and phrasing reproduce to the third
  decimal. Answers: key-fact recall 0.762 / 1.000 / 0.881 -> 0.810 / 1.000 /
  0.905, exactly the projection; mean grounding 0.759 -> 0.759; verify-ok
  1.000 -> 0.967 (`aa-13` fails the check); refusal scores unchanged. Red team
  18/21 -> 18/21 with a different third failure: `jb-01` now passes, `cite-01`
  refused correctly but repeated the forbidden `[S9]` label. Refusals gate /
  model 0/10. The gate has fired zero times over 61 questions run three times.

## 2026-10-24 — The cross-encoder reranker stays off by default (opt-in)

- **What:** `RERANKER=none` remains the default; the cross-encoder
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`, `rag/rerank.py`) stays an opt-in set
  with `RERANKER=cross_encoder`. It was measured this day on both corpora at
  k = 3 (`run_eval.py`): ai_act MRR 0.746 -> 0.891 with recall@3 0.957 -> 0.935;
  filings MRR 0.840 -> 0.940 with recall@3 / hit@3 0.900 -> 1.000.
- **Why:** three reasons outweigh the MRR gain. (1) It needs `torch` +
  `sentence-transformers` (`requirements-optional.txt`, hundreds of MB); the
  default install stays numpy-only, as the README advertises. (2) It re-scores
  `CANDIDATE_K` candidates on every query — a real per-query cost on CPU that a
  silent default should not impose on the interactive UI. (3) It is not strictly
  dominant: ai_act recall@3 dips (noise at n = 23, but not a free win), and
  recall is what feeds cite-or-refuse. The MRR gain is in ordering, while the
  answer receives the full `top_k` set regardless of internal order, so a better
  rank-1 helps less at answer time than the MRR delta suggests.
- **What changes in the numbers:** nothing in the published default tables — they
  are measured with `RERANKER=none`. The figures above are the opt-in path and
  are reported as such.
- **Recommended when:** `torch` is installed and the latency budget allows it
  (set `RERANKER=cross_encoder`). It is the strongest single lever on MRR and
  takes filings retrieval to recall@3 = 1.000.
- **Revisit if:** the default install gains torch for another reason (then the
  dependency objection falls), or a lighter reranker (ONNX / distilled) removes
  the latency cost — either would make a default-on reranker defensible.

## 2026-10-25 — Hugging Face joins the LLM fallback chain only when a token is set

- **What:** a `huggingface` provider in `rag/llm.py` calls Hugging Face's
  OpenAI-compatible router (`router.huggingface.co/v1`) with `HF_TOKEN` and
  `HF_MODEL` (default `meta-llama/Llama-3.1-8B-Instruct`). `LLM_PROVIDER=huggingface`
  makes it first; otherwise it sits before Ollama, and only when `HF_TOKEN` is set.
- **Why conditional:** the provider order and its models are part of the LLM cache
  key (§9.3 of the deep dive). Listing it unconditionally would have changed the
  key of every request, so every cached answer — including those behind the
  published evaluations — would have been re-paid on the next run. Without a
  token, the order and the key are exactly what they were
  (`tests/test_llm_huggingface.py::test_cache_keys_without_huggingface_are_unchanged`).
- **What changes in the numbers:** nothing published. The default chain, models
  and cache keys are unchanged; no evaluation was re-run with this provider.
- **Measured on the day:** the router accepts the account token but answers 402
  ("no remaining credits"): the account has no Inference Providers credits, so in
  practice the provider currently falls through to the next one.
- **Revisit if:** the account gets credits — then the answer eval can be run with
  `LLM_PROVIDER=huggingface` to compare it with Groq.
