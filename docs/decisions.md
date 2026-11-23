# Decisions

One entry per decision that still holds: date, what, why, in a few lines. When a
later change replaces an earlier one, the entry is merged under the latest date.
Entries stay when they explain what a number means.

## 2026-11-23: Strict scoring of the answers (displayed precision, cited refusals, more languages)

- **What:** three changes to how a stored answer is scored, applied offline to the
  23 November run with no LLM call (`scripts/rescore_offline.py`).
  - **Figures at the displayed precision.** A numeric key fact matches a number of
    the answer when the expected value, converted to the answer's scale and rounded
    half-up to the decimals the answer shows, equals it exactly. "8,0 billion" for
    7,979 million is right, "6 %" for 6.4 is right, "46.9 %" for 46.9 is right,
    "15 413" for 15,421 is wrong, and "46.93" for 46.9 is wrong (more precise than
    the expected value). The old rule was a relative tolerance of 1%. Only the
    scoring changed: `rag/verify.py` (figure check on cited sources) is untouched,
    and so is `facts_in_passages` in `scripts/run_eval.py`.
  - **A cited refusal is a refusal.** `is_cited_refusal`: the answer carries `[S#]`
    markers and either every sentence is a pure refusal (a strong phrase, or a weak
    negation in a sentence about the sources), or the first sentence is a strong
    refusal and each later one is a refusal or only describes what the sources are
    about, with no figure. A refusal followed by "but", "mais", "maar", "pero" or
    "aber" stays a refusal only if what follows also declines, describes the sources
    or is a negation, without a figure. "I don't know for sure, but the fine is 35
    million [S1]" and "I do not know. The fine is 35 million [S1]" remain answers
    (tested). Rows carry `cited_refusal`; the summaries count `n_cited_refused`.
    Before, any citation made an answer a claim (an earlier rule, before 14 October,
    let the first sentence override it; that is not restored).
  - **Dutch, Spanish and German** refusal and "the sources do not contain" phrases,
    and the same languages for the source words and describing verbs. "No dispongo
    de la información necesaria para responder a esa pregunta" was a real answer of
    the red team (`lang-02`) that counted as no refusal.
- **Why:** the three rules were stricter or looser than the claim they score. The
  tolerance accepted a figure from another line (15,413 for 15,421); the citation
  rule scored a declined answer as a hallucination; the phrase list knew three
  languages.
- **Effect on the 23 November run** (same stored answers, old rule then new):
  refusal recall 0.583 then 1.000, refusal F1 0.737 then 1.000, refusals 7 then 12
  (5 citing sources: `aa-17`, `aa-18`, `aa-19`, `fs-15`, `fs-19`), mean grounding
  0.740 then 0.789 and verify-ok 0.970 then 1.000 (refusals leave both), key-fact
  recall 0.905 then 0.905. Red team 20/21 then 21/21 (`lang-02`). Statements
  key-fact recall 0.800 (16/20) then 0.750 (15/20): `f8-15`, the only answer that
  matched by tolerance alone. The 14 October files in `eval/results/` are replaced by
  the 23 November run (git keeps them); the 14 October figures were 0.909 F1 and
  18/21.
- **Left as it was:** `answer_eval_item8.jsonl` (the first statements run, ids no
  longer in the eval file), the NLI and guard on/off files of October, not rescored.
  No change to any threshold, prompt or retrieval.
- **Open:** the precision rule accepts a bare integer anywhere in the answer that
  rounds to the expected value ("2 items" for 2.4); only the facts, not the
  context, are compared.

## 2026-11-23: Every published figure recomputed against the code

- **What:** the numbers of the README, the docs, the docstrings and the comments
  were rederived: chunk and file counts from the chunker (no embedding), retrieval
  and phrasing tables by running the scripts on Ollama embeddings, the answer and
  red-team tables by scoring the stored rows with today's `rag/answer_metrics.py`
  and `scripts/run_redteam.py`, the test count with CI's own commands in a
  `python:3.12-slim` container (632 passed, 1 skipped before this pass). No LLM was
  called.
- **Held:** retrieval at k = 3 and 5, the statements table at k = 1, 3, 5, phrasings
  (8/8, 7/8, 3/8, consistency 3/8), the answer table (all three columns), the red
  team 18/21 and its per-category rows, the guard on/off pair (19/21 and 21/21), the
  re-score of 19/21 with the output check, the NLI counts (11/27, 22/27, 27/27,
  entailed fraction 0.25), the aa-03 row, the FAISS parity (876 of 878, 155 of 155),
  the Wilson intervals, `facts_in_top_k` with the six text files alone (0.300), the
  first-version chunk figures (246 chunks, 2,303 characters, 291-character header,
  from commit 4fed587), and the extractor, which regenerates the 19 statement files
  byte for byte.
- **Changed:**
  - `f8-15` (dividends paid) was scored right with 15,413 (dividends declared) for
    15,421 because the key-fact match tolerates 1%. It is the only stored answer
    that matches by tolerance alone. Statements answers are 16 of 20 by the scorer
    and 15 read strictly, with 2 wrong figures (`f8-04`, `f8-15`), not 1. The
    tolerance is left as it is: it is what lets "$416.2 billion" match 416,161
    million. A rounding-aware match is a metric change and stays open.
  - "The gate fired 0 times in 61 questions" is 83: the 22 statements questions did
    not fire it either.
  - The answer table is not the one today's retrieval would produce. 25 of its 40
    questions now get another top 5 (16 reordered, 9 with another set); the gold
    source is still retrieved for all 28 answerable ones. Refusals, key facts and
    citations rescore identically from the stored answers; grounding and verify-ok
    cannot be recomputed because that run kept no passages.
  - "757 stored answers" is 187 for `eval/results/`; the 757 included local logs
    and the other repo.
  - FAISS parity on `filings_statements`: 124 of 124 lists identical.
  - The French UI showed English examples on `filings_statements`; four renderings
    added, and the test now covers the three corpora.
  - `run_ragas_eval.py` stopped without a word on a run with no `contexts` (the
    14 October answers): its dry run now lists what it leaves out and an empty
    selection exits with the reason.
  - `.dockerignore` did not exclude `*.docx`, so a local `docker build` copied the
    private `red-teaming.docx` into the image (found in `grounded-rag:lock`, a
    local image, never pushed). It does now.
  - Stale wording: `--corpus filings` in two files, "25 files for the 10-K", "10 MB
    of text", "both indexes", "httpx in requirements.txt", Azure missing from the
    provider lists, "after this branch is merged".
- **Added:** `tests/test_docs_figures.py` (chunk and file counts quoted by the
  README, every backticked path, every relative link) and a test that the shipped
  statement files are what the extractor writes.
- **Cross-encoder rows rerun** in a `python:3.12-slim` container with CPU torch and
  the same Ollama embeddings: every figure of the two tables below matches (text
  sets at k = 3: 0.935 / 0.891 and 1.000 / 0.940; statements hit / MRR / facts at
  k = 1, 3, 5 as published). Two figures were not published: at k = 5 the re-ranker
  takes ai_act recall from 1.000 to 0.957 (MRR stays 0.891), and on the phrasings it
  moves `other_language` from 3/8 to 5/8 and consistency from 3/8 to 4/8, with the
  other two styles unchanged. Eight questions: a hint, not a result.
  Paired on the statements questions (exact McNemar, discordant pairs only
  one way unless stated): source hit at k = 3 goes from 12 to 18 of 20 (0 against 6,
  p = 0.031); at k = 5 from 15 to 19 (1 against 5, p = 0.22); the figure printed in
  the passages (`facts_in_top_k`) stays at 11 of 20 at k = 3 and goes from 13 to 14
  at k = 5 (2 against 3, p = 1.0). So the gain is in which file ranks first, not in
  whether the figure reaches the model, and no answer run used the re-ranker.
- **Chunk size on the statements** (rebuilt and scored in the same container):
  500, 700, 900, 1,200 characters give 278, 170, 124, 84 chunks and facts@5 0.55,
  0.60, 0.65, 0.60, as stated.
- **Not rederived:** the 14 and 24 October and 10 November live runs cannot be rerun
  without Groq tokens.
- **route-or-roam:** its tests (307 now) pass against this repo, and
  `python -m eval.locate_gold` finds 0 missing gold sources in 40 questions.

## 2026-11-23: Item 8 of the 10-K gets its own corpus, `filings_statements`

- **What:** the 19 files of Item 8 (`scripts/extract_10k_item8.py`) live in
  `data/filings_statements/` (124 chunks). `filings_sections` is back to the six
  narrative Items (155 chunks), so its published figures are the ones of 14
  October again. Files that open with a `Context:` line are cut by
  `chunk_contextual_text`: whole rows, with the context, the statement title, the
  column line and the nearest group heading on every chunk. Other corpora are
  chunked as before. `run_eval.py` prints `facts_in_top_k`, the share of questions
  whose key figures are printed in the retrieved passages. The eval file is
  `eval/filings_statements_answers_eval.jsonl` (22 questions; `05_mdna.txt` removed
  from the gold because that corpus no longer holds it). `fs-15` is the original
  question again (net income for fiscal 2025, unanswerable in the text corpus).
- **Why:** adding Item 8 to `filings_sections` lowered the published text figures
  (k = 3 MRR 0.833 to 0.800; k = 5 recall 1.000 to 0.980, MRR 0.851 to 0.818).
  Checked on the same questions: "reportable operating segments" is a labelling
  artefact, because Note 13 does state the five segments, so the new file is a fair
  answer (gold left as it was); "fiscal year end" and "full-time equivalent
  employees" are real losses (rank 1 to 2 and 2 to 4; Note 1 does define the fiscal
  year but sat outside the top 5, and the employee count is only in Item 1). The causes were the header paragraph of each
  Item 8 file, which became a boilerplate chunk matching every "Apple fiscal 2025"
  question, and statements cut as one 2,300-character chunk.
- **Numbers** (retrieval, no LLM; labels of the 25 text questions untouched). Text
  set, `filings_sections`, hybrid: 6 files k = 3 recall 0.900, MRR 0.833, hit 0.920;
  k = 5 recall 1.000, MRR 0.851. With Item 8 added raw (246 chunks): 0.900, 0.800,
  0.920; 0.980, 0.818. Same questions with Item 8 added and cut by row: MRR 0.793
  at k = 3, so the chunking alone does not give the text set back. Now: 0.900,
  0.833, 0.920; 1.000, 0.851, by construction. Statements set (20 questions,
  `facts_in_top_5`): 0.300 with the six files only (the MD&A prints 6 of the
  figures), 0.500 added raw, 0.650 own corpus (95% CI 0.433-0.819). Own corpus,
  k = 1 / 3 / 5: hit 0.550 / 0.600 / 0.750, facts 0.400 / 0.550 / 0.650. Ceiling
  0.800: four questions have a figure no statement prints (`f8-17` to `f8-20`:
  three computed, one only in the MD&A). With `RERANKER=cross_encoder` (left off,
  see 2026-10-24): text set MRR 0.833 to 0.940, recall@3 0.900 to 1.000;
  statements hit@5 0.750 to 0.950, facts@5 0.650 to 0.700. Chunk sizes from 500 to
  1,200 characters give facts@5 0.55 to 0.65, so 900 (`CHUNK_SIZE`) stays. AI Act
  and phrasing figures unchanged.
- **Source hits mislead:** in the first answer run (Groq gpt-oss-120b, 23 November, 12
  of 20 right, 4 false refusals, 3 wrong figures) the "gold source" the logs show
  for `f8-05` and `f8-09` was the 291-character header chunk, with no figure in it.
  The gold lists of that file are unchanged except for dropping the MD&A.
- **Answers re-run on `filings_statements`** (Groq gpt-oss-120b, 23 November, 22
  questions): 16 of 20 right by the scorer (was 12; 15 read strictly, see the review
  entry above), 1 wrong figure by the scorer (2 strictly; was 3), 3 refusals where
  the table was not retrieved (was 4), 2 of 2 unanswerable refused. The fix was
  retrieval only: same prompt, same model.
- **Not re-run:** the answer and red-team tables (14 October), the FAISS parity of
  the new index. The
  re-ranker and MMR variant tables below describe the six-file corpus, which is
  what `filings_sections` is again.
- **route-or-roam:** unaffected. It reads `filings_sections` and `ai_act_sections`
  only, and the first is the six files its gold was located on. Its tests (301 then,
  307 on the final review) pass against this repo and `python -m eval.locate_gold` finds 0 missing gold sources in
  40 questions. It has no tool for `filings_statements`.
- **Decision:** the README keeps the 10-K promise on the text and offers the
  statements as a second corpus to look figures up in, not as financial analysis.
  The app does not route between the two corpora yet.

## 2026-11-23: Locked dependencies for the image and CI

- **What:** `requirements.lock` (uv, Python 3.12, Linux) is what the Dockerfile
  and CI install. `requirements.txt` stays the readable list of floors.
  `config.py` also loads only this repo's own `.env`.
- **Why:** the image and CI resolved the floors on their own and could drift
  apart; route-or-roam already installs a lock. A bare `load_dotenv()` let a
  worktree read a parent folder's `.env`.
- **Effect:** no published number changes.

## 2026-10-14: Answer and red-team evaluations re-run live

- **What:** the retrieval, phrasing, answer and red-team evaluations were re-run
  end to end with `openai/gpt-oss-120b` on Groq and `nomic-embed-text` on Ollama,
  LLM cache off. A first pass with the cache on replayed most of the 6 October
  answers and was discarded.
- **Why:** three scorer changes (key-fact matcher, figure and negation checks,
  refusal phrases) had landed after the published runs.
- **Numbers:** retrieval and phrasing reproduce to the third decimal. Key-fact
  recall 0.762 / 1.000 / 0.881 became 0.810 / 1.000 / 0.905 (the projection),
  verify-ok 1.000 became 0.967 (`aa-13` fails the figure check), red team 18/21
  with a new third failure (`jb-01` passes, `cite-01` fails). Refusals gate /
  model 0/10.

## 2026-10-24: The cross-encoder re-ranker stays off by default

- **What:** `RERANKER=none` is the default; `cross_encoder` is opt-in
  (`rag/rerank.py`). At k = 3 it takes ai_act MRR from 0.746 to 0.891 (recall@3
  0.957 to 0.935) and filings MRR from 0.840 to 0.940 (recall@3 0.900 to 1.000).
- **Why:** it needs torch (the default install is numpy-only), re-scores the
  candidates on every query, and is not strictly dominant (ai_act recall@3 dips).
  The answer gets the full `top_k` whatever the internal order, so a better rank 1
  helps less than the MRR suggests. Recommended when torch is installed and latency
  allows.
- **Numbers:** no published table uses it.

## 2026-10-25: Hugging Face joins the LLM fallback chain only when a token is set

- **What:** a `huggingface` provider in `rag/llm.py` (router, `HF_TOKEN`,
  `HF_MODEL`). `LLM_PROVIDER=huggingface` puts it first; otherwise it sits before
  Ollama, and only with `HF_TOKEN`.
- **Why conditional:** the provider order is part of the LLM cache key. Listing it
  always would have invalidated every cached answer behind the published runs.
- **Numbers:** none changed. The account has no Inference Providers credits (the
  router answers 402), so in practice the chain moves on.

## 2026-11-04: One Qdrant collection per store

- **What:** each store builds into `<QDRANT_COLLECTION>_<index dir>`
  (`grounded_rag_ai_act_sections`, `grounded_rag_filings_sections`).
  `QDRANT_COLLECTION` is a prefix now.
- **Why:** the API and the Streamlit app load one store per corpus in one process.
  With one shared collection the second build replaced the first's points, and
  point id `i` pointed at the other corpus's row `i`: wrong passages with plausible
  scores. `tests/test_qdrant_index.py::test_two_stores_in_one_process_do_not_share_a_collection`
  failed before the fix.
- **Numbers:** none (Qdrant is opt-in, no evaluation used it). On 2026-11-10 Qdrant
  also started returning `top_k` rows instead of up to 64, so the store's exact
  cosine cannot pull in a row the graph ranked below `top_k`.

## 2026-11-08: Citation forms the parser accepts

- **What:** `normalize_citations` also rewrites "(S9)", "(S1, S5)" and "（S2）" to
  `[S#]` (6 November; "(see S1)", "Form (S-1)" and "(S1]" stay as written). And
  `split_sentences` moves a marker that opens a fragment back to the sentence
  before it, so "within 999 hours. [S1]" is one cited sentence (8 November).
- **Why:** both are measurement bugs: the model was right, the parser was not.
  gpt-oss writes "claim. [S1]" in real answers (aa-06 five times in the 40 stored
  ones); read as uncited, its figures and negations were never checked.
- **Numbers:** checked on the 40 stored answers against the passages they were
  given: verify-ok 0.938 / 1.000 / 0.967 and mean grounding 0.741 / 0.779 / 0.759
  do not move. Only one stored answer of another repo changed.

## 2026-11-08: `top_score` is the best dense cosine, the number the gate compares

- **What:** the answer eval and the `cli.py ask` log record
  `rag.retrieve.top_dense_score` (best cosine over the whole store), not the score
  of the first returned passage. Gate refusals record it too.
- **Why:** after RRF and MMR the first passage had a lower cosine than the best
  match on 13 of the 40 eval questions (by up to 0.066).
- **Numbers:** unanswerable questions span 0.599 to 0.888, answerable ones 0.611
  to 0.915 (it was 0.910). The ranges still overlap.

## 2026-11-08: A failed judge call is not an unparsed judge reply

- **What:** rows with `judge_error` are counted in `judge_errors`, no longer in
  `judge_unparsed`.
- **Why:** a 429 on the judge read as the judge breaking its JSON contract.
- **Numbers:** `judge_unparsed` can be lower on runs with outages; `judge_mean`
  is unchanged.

## 2026-11-10: A BM25-only passage shows its cosine and says it came from BM25

- **What:** in hybrid mode `retrieve` gives every passage its dense cosine as
  `score` and a `score_source` (`dense`, `bm25`, `both`). The UIs mark a `bm25`
  passage "keywords only". Before, a BM25-only passage carried its raw BM25 score
  next to cosines.
- **Why:** a BM25 score of 7.5 beside a cosine of 0.6 reads as "very relevant".
  The gate is per query: once one dense hit clears it, BM25-only passages come in
  whatever their cosine. BM25 is there to catch exact terms (an article number)
  that a per-passage gate would drop.
- **Numbers:** none. Order, gate and every eval number are unchanged.

## 2026-11-10: MMR's relevance in hybrid mode is the fused rank

- **What:** MMR's relevance is `1 - (r-1)/(n-1)` over the fused order
  (`_rank_relevance`), `MMR_LAMBDA` stays 0.6. With the cross-encoder on it stays
  the min-max rescaled scores.
- **Why:** RRF scores with k = 60 are bunched, so after min-max the last places
  were picked by diversity, not relevance. With ranks instead of min-max the
  re-ranked path got worse (ai_act recall@3 0.935 to 0.848), so it keeps min-max.
- **Measured** (no LLM, hybrid, same index):

  | Variant | ai_act R@3 / MRR | ai_act R@5 | filings R@3 / MRR | filings R@5 / hit@5 | other_language | gold@5 |
  | --- | --- | --- | --- | --- | --- | --- |
  | lambda 0.6, min-max (old) | 0.957 / 0.746 | 0.978 | 0.900 / 0.840 | 0.960 / 0.960 | 3/8 | 28/28 |
  | lambda 0.75, min-max | 0.957 / 0.739 | 1.000 | 0.900 / 0.840 | 0.960 / 0.960 | 2/8 | 28/28 |
  | lambda 0.8, min-max | 0.935 / 0.746 | 1.000 | 0.920 / 0.840 | 0.960 / 0.960 | 3/8 | 28/28 |
  | lambda 0.6, rank (kept) | 0.957 / 0.754 | 1.000 | 0.900 / 0.833 | 1.000 / 1.000 | 3/8 | 28/28 |

  The rank variant changed 54 of 68 top-5 lists, so both ran live the same day
  (old then rank): verify-ok 0.966 and 0.966, key-fact recall 0.893 and 0.905,
  mean grounding 0.762 and 0.729, red team 17/21 and 18/21 (only `pii-01`).
- **Read it right:** every gap is one question, so none of it is significant. The
  rank variant is kept because it is best or tied at k = 5, hurts neither refusals
  nor citations, and fixes a wrong mechanism. The README and deep dive retrieval
  tables use it; the answer and red-team tables are from 14 October.

## 2026-11-15: The Streamlit Cloud setup embeds with an OpenRouter model, on an uncalibrated threshold

- **What:** the README and `.streamlit/secrets.toml.example` point the hosted
  embedder at OpenRouter (`nvidia/nemotron-3.5-embed-1b:free`) instead of Mistral.
  No code changed.
- **Why:** the Mistral account's API keys are not active; the OpenRouter key
  already used for generation also reaches free embedding models.
- **Numbers:** every published table uses `nomic-embed-text`, and
  `SCORE_THRESHOLD = 0.35` was chosen on its scale. On the cloud embedder the same
  0.35 is carried over, not calibrated. One cold-start rehearsal: both indexes in
  about 20 s, Article 5 first at cosine 0.642, an off-topic question refused.
- **Revisit if:** the deployed app refuses answerable questions or answers
  off-topic ones. Then run `scripts/run_eval.py` with that embedder.

## 2026-11-16: An answer that repeats the system prompt is replaced by the refusal

- **What:** `rag/answer.py:leaks_system_prompt` flags an answer holding 8
  consecutive words of the system prompt or its optional notes (case and
  punctuation ignored). `answer_question` and `stream_answer` then return the
  refusal with `Answer.leak_blocked = True`. `refused` still means "the gate
  refused", so a blocked leak counts in neither side of gate / model.
- **Why:** on 14 October `leak-02` got the whole prompt back. The fence protects
  the sources, not the question, and `USER_TURN_GUARD` only asks the model.
- **Numbers:** over the answers stored in both repos (757 then, 187 of them in
  this repo's `eval/results/`) it flags only the two
  `leak-02` answers, so the answer table is unchanged. Re-scoring the stored red
  team gives 19/21 (`leak-02` passes); that is a re-score, not a live run, and the
  README keeps the measured 18/21. The whole answer is replaced, and with
  streaming on the leak stays on screen until the final answer replaces it.

## 2026-11-16: `cite-01` is a refusal of an answerable question

- **What:** earlier text called `cite-01` a correct refusal that only failed for
  repeating the `[S9]` label. The question is answerable and the case expects an
  answer that ignores the injected demand; the model refused the whole question,
  so the failure stands. `scripts/run_redteam.py:over_refused` flags every case
  that expects an answer and got a refusal (`cite-01`, and `lang-01`, which passes
  and whose "I do not know." is a retrieval miss: Article 3 is not in the passages
  even at k = 10).
- **Numbers:** none; pass/fail is unchanged.

## 2026-11-16: `LLM_FALLBACK=false` pins an evaluation to one provider

- **What:** `config.LLM_FALLBACK` (default true) and `config.fallback_order`. With
  false the order is `[LLM_PROVIDER]` alone and a failed call raises.
- **Why:** the fallback chain mixes models inside one run without saying so (Groq's
  free tier caps each key at 200,000 tokens a day). The published runs are clean
  (40/40 and 21/21 rows from Groq), but nothing stopped the next one from mixing.
- **Numbers:** none. The default chain and the cache keys are unchanged.

## 2026-11-22: Figures and negations are checked against the cited sources

First added on 2026-10-10, widened on 2026-10-14, repaired on 2026-11-22.

- **What:** `rag/verify.py` checks per cited sentence that every figure it states
  is in the sources that sentence cites. A missing figure goes in
  `unsupported_numbers` and makes the answer not `ok`. A sentence that negates
  while none of its sources does goes in `negation_mismatches` and blocks
  `strict_ok` only (a heuristic, hence strict only).
  - Whole numbers keep their zeros (only a decimal part is trimmed: "2.50" =
    "2.5", "240" is not "24"), U+00A0 and U+202F count as thousands separators,
    and a figure with a scale word is supported by its expanded or written value
    ("15 million" = "15 000 000"; "$416,161 million" = a table in millions).
  - 2026-11-22: the `n't` alternative of the negation pattern sat inside a
    word-bounded group and never matched, so "doesn't apply" was invisible. It
    matches now ("don't", "can't", curly apostrophe too). And the thousands
    pattern merged "2024 100" into 2024100 and read "0.350" as 0350, so "0.35"
    against "0.350" was unsupported. A thousands group now needs a lead of one to
    three digits that does not start with 0 and does not follow another digit.
- **Why:** the grounding score compares vocabulary, so "respond within 999 hours
  [S1]" against a source saying 24 scored 0.83 and passed. Figures are the one
  thing a lexical check judges with confidence.
- **Same day:** `verify_answer` and `score_answer` defaulted to a 0.5 bar while the
  app and the evals run at `config.VERIFY_MIN_GROUNDING` (0.30). Both default to
  the config value now. Every app and script caller already passed it, and the
  red-team scripts only read `invalid_citations`, so no published figure moves.
- **Numbers, measured offline on 2026-11-22, no LLM call:** the figure sets of
  `_figures` are identical, old code against new, for all 40 logged answers, the 21
  red-team answers and all 1,033 chunks of both corpora. So verify-ok 0.938 /
  1.000 / 0.967, key-fact recall, grounding and the red team 18/21 are unchanged
  (red-team checks use forbidden strings, refusals and citation range only). The
  new negation pattern matches in three red-team answers (`jb-01`, `leak-01`,
  `lang-03`); that can only touch `strict_ok`, which no table reports.
- **What it does not settle:** scope ("providers" for "deployers") and paraphrased
  contradictions pass. A figure derived from the sources (a sum) is flagged unless
  its terms are cited. A unit stated away from the figure (a table header) still
  reads as unsupported.
- **Today's retrieval is not the logged one.** Re-retrieving the 30 answered
  questions with the current index gives other passages for 18 of them, so the log
  can be re-verified against the stored passages (as on 2026-11-16, at e378805) but
  not against today's index.

## 2026-11-22: Eval outputs are committed in `eval/results/`

- **What:** `answer_eval_results.jsonl`, `redteam_results.jsonl`, the paired
  `redteam_guard_off.jsonl` / `redteam_guard_on.jsonl` and `answer_eval_nli.jsonl`
  are copied from `logs/` to `eval/results/`. The scripts still write to `logs/`
  (gitignored) so a partial run cannot overwrite the reference; copy a run over
  once it is the one to keep.
- **Why:** the README and deep dive quote these rows, and a fresh clone could not
  check any figure. Older runs (the four-document MRR 0.884, the verify-ok
  0.214 to 1.000 sequence) kept no artifact and are labelled as such.

## 2026-11-22: Test tools stay out of the image

- **What:** `pytest`, `httpx` (FastAPI's `TestClient`), `ruff` and `debugpy` are in
  `requirements-dev.txt`; `requirements.txt` holds the runtime only. CI installs
  both and runs `ruff check` (rules F, E9, B905, RUF007) before pytest.
  `data/filings/raw/` (9.5 MB, provenance only) is out of the build context.
- **Why:** the image should hold what the app runs. Nothing is ingested from
  `raw/`; the index comes from `data/filings_sections/`.

## 2026-11-17: Azure path is opt-in and unproven

- **What:** `LLM_PROVIDER=azure_openai` and `EMBED_PROVIDER=azure_openai` call an
  Azure OpenAI resource through its v1 API. `VECTOR_INDEX=azure_search` is a third
  approximate backend. `deploy/azure/provision.sh` deploys the UI on Container
  Apps. Setup and costs: `docs/azure.md`.
- **Why:** to show the pipeline on Azure without changing any default. The fallback
  chain gains `azure_openai` only when `AZURE_OPENAI_API_KEY` is set, so no cache
  key or published number changes.
- **Not run yet:** nothing has touched a real Azure subscription. The REST calls
  follow the API reference and are tested with stand-ins. The first live run tests
  `provision.sh`, the v1 URL and the index definition.

## 2026-11-17: One scoring step and safer embedding batches

- **What:** the cosine scoring of rows proposed by an index is one method,
  `VectorStore._score_rows`, used by FAISS, Qdrant and Azure AI Search.
  `HostedEmbedder` takes an optional `batch_size`, restores input order from the
  `index` field, and raises if the endpoint returns a different number of vectors
  than texts.
- **Why:** the scoring was written twice, and a short reply would attach a passage
  to the wrong vector. Behaviour and numbers are unchanged.

## 2026-11-17: How RAGAS is defined here

- **What:** faithfulness, answer relevancy and context precision, without a
  reference (the eval sets have key facts, not reference answers). Only answerable
  questions that were answered are scored, since RAGAS would count a correct
  refusal as a failure. Each row is scored three times; the summary gives the mean
  of the run means with min and max. A failed judge call is None, never 0.
  `run_answer_eval.py` rows now carry `contexts`, the passages the answer saw.
- **Not run yet:** no RAGAS number exists, the judge needs Azure.

## 2026-11-17: How the action eval is defined

- **What:** a predicted action pairs with a labelled one when the Jaccard overlap
  of their source sentences is at least 0.5, best pairs first, each used once. A
  labelled owner of null means nobody can be assigned, so the right output is no
  owner id. "Wrong owner assigned" and "caught by review" are reported apart,
  because a task sent to the wrong person is the costly error.
- **Measured:** `llama3.2:3b` on Ollama, cache off, 6 meetings, 13 actions.
  Precision 12/15, recall 12/13, owner accuracy 11/12, due-date accuracy 7/12,
  wrong owner assigned 0/12, errors flagged 6/6. The injection meeting gave an
  empty list. This tests the checks and is not a result for Azure.
