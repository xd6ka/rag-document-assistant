# Answer audit — 29 questions, local Qwen2.5-3B

Two audited runs are recorded here: the original MMR-only run (`generated_at` 2026-11-20) and a
re-run after the cross-encoder became the default stage (`generated_at` 2026-11-21). Both used the
same model and the same 29 answers-worth-of-time; the second run is what `latest.json` now holds.

Model: `ollama · qwen2.5:3b`, temperature 0.2, default retrieval (800/150, retrieve 8 → MMR → k=4
for run 1; 800/150, retrieve 8 → cross-encoder → MMR → k=4 for run 2).
Each answer below was checked against the actual gold page text, not against the model's own wording.

Why a manual audit exists at all: an LLM judge was tried first and rejected — see
`eval/reports/judge_experiment.json`, where the same 3B model scored seven grounded answers as
1/5. Its space is preserved as evidence, not as a metric.

## Run 2 — with the cross-encoder (what `latest.json` holds now)

Retrieval: hit@k 1.00, hit@1 0.96, MRR 0.980. Answer level: keyword 24/25, citation rate 25/25,
**citation gold rate 20/25**, **claim support 0.80**, refusal 3/4.

Every in-scope answer is substantively correct on this run — **manual accuracy 25/25 = 1.00**.
Two changes from run 1:

- **`p04` is fixed.** With the cross-encoder the gold page (`prompt-design.pdf` p3) arrives as block
  `[1]` and the answer names citation drift instead of "contamination". Keyword accuracy rises
  0.92 → 0.96, and the only lexical miss left is `d04` (the "annotation" paraphrase).
- **The remaining defect moved to citation attribution.** Five answers cite the wrong block while
  being right about the fact. In all five the gold page was block `[1]`, ranked first:

  | id | gold | cited block | cited page | answer correct? |
  |---|---|---|---|---|
  | e05 | embedding-models p1 | `[2]` | embedding-models p2 | ✅ |
  | v02 | vector-store-notes p1 | `[3]` | vector-store-notes p3 | ✅ |
  | c03 | chunking-strategies p2 | `[3]` | retrieval-eval p3 | ✅ (cites another *document*) |
  | p04 | prompt-design p3 | `[2]` | embedding-models p2 | ✅ |
  | d04 | pdf-internals p3 | `[2]` | pdf-internals p1 | ✅ |

  This is why `citation_gold_rate` now exists alongside `citation_rate`: the presence check says
  1.00, the gold-page check says 0.80, and the gap is entirely generation-side. `c03` is the
  sharpest case — the model quotes chunking guidance and cites a retrieval-evaluation page.

- **One refusal regressed.** `o03` (DashScope price per million tokens) answered "all-MiniLM-L6-v2 has
  22 million parameters [1]" instead of declining. Run 1 refused all four out-of-scope questions with
  the same prompt and settings, so this is model variance, not a detector bug: 3/4 is reported.
  The five drifts above appeared identically in both runs, which is why they read as a property of
  the prompt rather than a lucky sample.

## Claim-level check against this audit

`src/citations.py` splits each answer into sentences, attaches the `[n]` each sentence cites (including
the common "… another page. [2]" shape, where the marker arrives after the final period), and asks
whether the cited block is the best match for that sentence — lexically and by embedding — against
the other blocks in the prompt. `eval/claim_audit.py` replays the retrieval and grades the stored
answers without generating anything; per-sentence verdicts are in `eval/reports/claims.json`.

Result on run 2: **mean claim support 0.80**, and the flagged sentences are exactly the five rows in
the table above — `e05`, `v02`, `c03`, `p04`, `d04`. None of the twenty correct answers was flagged
(false positives: 0). The automatic check therefore agrees with this hand audit 25/25, which is the
only reason it is reported as a metric rather than as a curiosity.

Two things it does **not** claim:
- it is an attribution check, not entailment: a sentence can name the right block and then assert
  something the block does not say;
- thresholds (0.06 embedding margin, 0.35 lexical-overlap floor) were chosen on these 25 answers, so
  the calibration is real but narrow — a different corpus needs its own audit before the number is
  trusted.

## Run 1 — MMR only (`generated_at` 2026-11-20)

### Run 1 in-scope questions (25)

| id | question | gold | verdict | note |
|---|---|---|---|---|
| e01 | parameters of all-MiniLM-L6-v2 | embedding-models p1 | ✅ 22M | verbatim match |
| e02 | MTEB retrieval of bge-small | embedding-models p2 | ✅ 62.2 | |
| e03 | family needing the query prefix | embedding-models p3 | ✅ e5 / `query: ` | |
| e04 | MiniLM weight memory | embedding-models p2 | ✅ 90 MB | |
| e05 | dimensionality of bge-base | embedding-models p1 | ✅ 768 | cites `[2]` while the fact sits in block 1 — correct answer, wrong bracket |
| v01 | HNSW neighbours per node | vector-store-notes p2 | ✅ 16 | |
| v02 | store keeping vectors next to relational data | vector-store-notes p1 | ✅ pgvector | |
| v03 | why two writers must not index at once | vector-store-notes p3 | ✅ file lock / not supported | repeats itself, but the fact is right |
| v04 | cosine space requirement | vector-store-notes p2 | ✅ normalized vectors | |
| c01 | answers lost across a cut without overlap | chunking-strategies p2 | ✅ 12% | |
| c02 | overlap used by this project | chunking-strategies p2 | ✅ 150 chars | |
| c03 | tail-merge threshold | chunking-strategies p2 | ✅ 80 chars | |
| c04 | what table questions lose | chunking-strategies p3 | ✅ columns | retrieved at rank 2 |
| r01 | MRR of a correct answer at position 3 | retrieval-eval p1 | ✅ 0.33 | |
| r02 | how many questions to write | retrieval-eval p2 | ✅ 30–40 | |
| r03 | hit@3 on a 4-chunk corpus | retrieval-eval p3 | ✅ 1.0 | |
| r04 | metric hiding the rank | retrieval-eval p1 | ✅ hit@k | |
| p01 | temperature for factual answers | prompt-design p1 | ✅ 0.2 | |
| p02 | tokens in four 800-char blocks | prompt-design p2 | ✅ ~700 | retrieved at rank 3 |
| p03 | effect hurting mid-prompt facts | prompt-design p3 | ✅ context dilution | |
| p04 | claim-by-claim vs rate-based metrics | prompt-design p3 | ❌ **wrong fact** | answered "contamination" instead of "citation drift"; the gold page never entered the top 4 |
| d01 | how text is stored in a PDF | pdf-internals p1 | ✅ character codes + font | |
| d02 | JPEG filter | pdf-internals p3 | ✅ DCTDecode | |
| d03 | JPEG pixel size | pdf-internals p3 | ✅ 480×240 | |
| d04 | what a link is technically | pdf-internals p3 | ✅ paraphrase | describes "clickable rectangle with an action" but omits the word **annotation**; the lexical metric counts this as a miss |

**Manual accuracy: 24/25 = 0.96.** The lexical `must_contain` metric reports 23/25 = 0.92; the gap is
d04, where the answer is correct in substance and only differs in wording.

### Run 1 out-of-scope questions (4)

| id | question | verdict |
|---|---|---|
| o01 | capital of Mongolia | ✅ refused ("I do not know [1]") |
| o02 | 2018 World Cup winner | ✅ refused ("does not contain information") |
| o03 | DashScope price per million tokens | ✅ refused ("does not provide information in the given context") |
| o04 | Rust borrow checker | ✅ refused |

`refusal_rate = 1.0`. An earlier run reported 0.75 only because o03's phrasing was missing from the
marker list — a detector bug, not a model failure. The detector now combines markers with a negation
regex (`tests/test_refusal.py`), and `--rescore` re-derives answer metrics from stored answers so
metric fixes do not require regenerating 29 answers on CPU.

## What this audit does not cover

- **Claim-level citation correctness.** `citation_gold_rate = 0.80` checks that the cited block's
  page is the gold page, not that every sentence in the answer is supported by it. An answer that
  cites the right page and then adds a sentence from a different block still passes.
- **Russian answers.** The question set is English; the multilingual behaviour of this 3B model is
  untested here even though the UI advertises it.
- **Latency under load.** 46.7 s mean per question on this machine (retrieval + generation),
  60.6 s max. Fine for an audit, too slow to feel interactive.
