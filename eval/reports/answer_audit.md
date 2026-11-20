# Answer audit — 29 questions, local Qwen2.5-3B

Model: `ollama · qwen2.5:3b`, temperature 0.2, default retrieval (800/150, retrieve 8 → MMR → k=4).
Each answer below was checked against the actual gold page text, not against the model's own wording.
Run that produced the answers: `eval/reports/latest.json` (`generated_at` 2026-11-20).

Why a manual audit exists at all: an LLM judge was tried first and rejected — see
`eval/reports/judge_experiment.json`, where the same 3B model scored seven grounded answers as
1/5. Its space is preserved as evidence, not as a metric.

## In-scope questions (25)

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

## Out-of-scope questions (4)

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

- **Citation correctness.** `citation_rate = 1.0` means every answer contains at least one `[n]`,
  not that `[n]` points at the block that supports the claim. e05 proves the difference: right
  answer, wrong bracket. Claim-level citation checking is the next metric to build.
- **Russian answers.** The question set is English; the multilingual behaviour of this 3B model is
  untested here even though the UI advertises it.
- **Latency under load.** 46.7 s mean per question on this machine (retrieval + generation),
  60.6 s max. Fine for an audit, too slow to feel interactive.
