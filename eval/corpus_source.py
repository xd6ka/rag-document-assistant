"""Source content for the evaluation corpus.

Six short documents, three pages each, written so that retrieval is genuinely
hard rather than decorative:

* every document has a "Limitations" or "notes" page with deliberately similar
  wording, so lexical matching alone tends to pick the wrong document;
* several documents talk about the same concepts (chunk size, overlap, top-k,
  citations) and only the numbers differ;
* facts are pinned to specific pages, which is what the citation metric needs.

The corpus is synthetic and the numbers are internally consistent but invented.
It is a test fixture for the pipeline, not a benchmark of real-world quality.
"""

from __future__ import annotations

DOCS: dict[str, dict] = {
    "embedding-models": {
        "title": "Local Embedding Models: A Comparison",
        "pages": [
            [
                "Three small models dominate CPU-friendly retrieval: all-MiniLM-L6-v2, "
                "e5-small-v2 and bge-small-en-v1.5. All three produce 384-dimensional "
                "vectors, which is convenient because they can share one index layout.",
                "all-MiniLM-L6-v2 is a 22-million-parameter sentence-transformer. It is the "
                "default in most tutorials because it downloads in about 90 megabytes and "
                "encodes roughly 900 short chunks per second on a laptop CPU.",
                "bge-base-en-v1.5 produces 768-dimensional vectors from a "
                "109-million-parameter encoder. It is slower, and it consistently scores "
                "higher on retrieval benchmarks.",
            ],
            [
                "The models are not equal on the MTEB retrieval average: "
                "all-MiniLM-L6-v2 reaches 56.0, e5-small-v2 reaches 59.8 and "
                "bge-small-en-v1.5 reaches 62.2. bge-base-en-v1.5 gains another three "
                "points but costs roughly four times the encoding time.",
                "Memory matters more than people expect. MiniLM needs about 90 megabytes "
                "of weights, while bge-base needs about 440 megabytes. That is fine on a "
                "laptop, but it is noticeable when an app keeps the model resident between "
                "requests.",
                "A practical rule: start with MiniLM to get the pipeline working, then swap "
                "the model behind the same interface and measure the difference on your own "
                "evaluation set instead of trusting a leaderboard.",
            ],
            [
                "All three models are trained primarily on English text. Multilingual input "
                "degrades quickly and should use a multilingual checkpoint instead.",
                "The e5 family expects an asymmetric prefix: documents are embedded as they "
                "are, while queries must be prefixed with the literal string 'query: '. "
                "Skipping this prefix costs several points of recall and is the most common "
                "mistake when swapping models.",
                "MiniLM needs no prefix. Mixing the two conventions silently produces an "
                "index that builds fine and retrieves worse.",
            ],
        ],
    },
    "vector-store-notes": {
        "title": "Vector Stores for a Single-Machine Pipeline",
        "pages": [
            [
                "Three options cover almost every small project: ChromaDB, FAISS and "
                "pgvector.",
                "ChromaDB ships a PersistentClient that writes to a local SQLite file plus "
                "binary index segments. It handles metadata filtering and document storage in "
                "the same call, which is why it is the easiest choice for a prototype.",
                "FAISS is a library, not a service: an index is a file you build and load "
                "yourself. It is faster and more flexible, but filtering, stored text and "
                "persistence become your responsibility.",
                "pgvector keeps vectors next to relational data, which pays off when the "
                "documents already live in PostgreSQL.",
            ],
            [
                "ChromaDB builds HNSW graphs underneath. The defaults in recent versions are "
                "M=16 neighbours per node and ef_construction=200. The ef_search value is "
                "tuned at query time and defaults to a setting that favours recall.",
                "Distance space matters more than the algorithm. Cosine space requires "
                "normalized vectors; with L2 you must keep the vectors normalized yourself or "
                "similarities drift. This pipeline stores normalized embeddings and declares "
                "cosine space explicitly.",
                "A collection is the unit of isolation. One collection per tenant keeps "
                "filters cheap; a single collection with a source metadata field is cheaper to "
                "operate for a handful of documents.",
            ],
            [
                "Persistence is one file lock away from trouble: two writers on the same "
                "directory are not supported, so a command-line script and a running app should "
                "not index at the same time.",
                "Deleting by metadata filter is supported but not free. Deleting a whole "
                "document and re-inserting it is usually the simplest way to keep an index "
                "consistent.",
                "Backups are trivial: copy the directory while nothing is writing to it, and "
                "the index comes back byte for byte.",
            ],
        ],
    },
    "chunking-strategies": {
        "title": "Chunking Strategies for Retrieval",
        "pages": [
            [
                "Three families of chunking are used in practice: fixed windows, "
                "sentence-aware splitting and semantic splitting.",
                "Fixed windows are the simplest: cut every N characters, optionally "
                "overlapping. They are fast, deterministic and easy to unit-test, but they can "
                "cut a sentence in half.",
                "Sentence-aware splitting cuts on punctuation and keeps clauses intact. It "
                "produces cleaner chunks but makes length variable, which complicates batching.",
                "Semantic splitting embeds sentences and starts a new chunk when the topic "
                "shifts. It is the most expensive and the least predictable of the three.",
            ],
            [
                "Overlap exists to defeat the boundary problem. With no overlap, roughly 12 "
                "percent of answers in a paragraph-level evaluation sit across a cut and become "
                "unreachable.",
                "A 15 to 20 percent overlap is the usual compromise. This project uses "
                "800-character windows with 150 characters of overlap, which is about 19 "
                "percent.",
                "Very short chunks are worse than no chunk: a 40-character fragment carries "
                "almost no retrievable meaning. Merging tails shorter than 80 characters into "
                "the previous chunk removes most of the junk.",
            ],
            [
                "Page-scoped chunking costs a little context on pages that continue "
                "mid-sentence, but it guarantees that every citation carries a page number. "
                "Trustworthy citations were worth more here than marginal context.",
                "Tables remain the weak spot: a table is text positioned in a grid, and naive "
                "extraction interleaves rows and columns, so table questions often retrieve the "
                "right page and lose the columns.",
                "Chunking cannot fix a bad extractor. If the page text is empty, chunking only "
                "distributes emptiness across more vectors.",
            ],
        ],
    },
    "retrieval-eval": {
        "title": "Evaluating Retrieval and Answers",
        "pages": [
            [
                "Retrieval quality is measured with labelled questions: each question has one "
                "or more gold chunks or pages, and the metric asks whether the gold item "
                "appeared in the returned list.",
                "hit@k counts a question as solved if any gold item is in the top k. It is the "
                "most quoted and least informative metric, because it hides how far down the "
                "answer was.",
                "MRR, mean reciprocal rank, averages 1/rank of the first correct item: a "
                "correct answer at position 1 scores 1.0, at position 3 it scores 0.33. nDCG "
                "adds graded relevance and is overkill for single-answer questions.",
            ],
            [
                "Gold labelling should be mechanical rather than intuitive: record the page, "
                "then verify that the page text actually contains the fact the question asks "
                "about. A validation step that fails loudly catches rubric drift early.",
                "Mix question types: single-fact lookups, comparison questions that span two "
                "documents, and out-of-scope questions the corpus cannot answer. The last type "
                "is the only direct test of grounding.",
                "Thirty to forty questions over a corpus of thirty chunks is enough to see a "
                "difference between configurations. Five questions are not.",
            ],
            [
                "Saturated metrics are the first trap: with four chunks in the corpus, hit@3 "
                "is 1.0 for almost any configuration, which proves the wiring works and says "
                "nothing about quality.",
                "Lenient judges are the second trap: asking the same model that generated an "
                "answer to grade it produces inflated scores. Report such a judge as a sanity "
                "check, never as an independent measurement.",
                "Contamination is the third trap: questions written after looking at the "
                "retrieved chunks tend to encode the retriever's own mistakes. Write questions "
                "from the source document, not from the search results.",
            ],
        ],
    },
    "prompt-design": {
        "title": "Prompt Design for Grounded Answering",
        "pages": [
            [
                "A grounded prompt has three parts: the numbered context, the question, and "
                "an explicit instruction about what to do when the context does not contain the "
                "answer.",
                "The instruction is the part people skip. Without it a model smoothly answers "
                "from its own parameters, which is indistinguishable from a correct answer until "
                "someone checks the page.",
                "Temperature matters for factual answering. A value of 0.2 keeps phrasing "
                "stable between runs; 0.7 belongs to creative writing, not to citations.",
            ],
            [
                "Numbering the context as [1], [2], [3] gives the model a cheap way to point "
                "at evidence, and gives the reviewer a one-click way to check the claim.",
                "Blocks should carry their source and page inside the block rather than in a "
                "separate legend, so the model can copy the reference without inventing an "
                "index.",
                "Four blocks of 800 characters come to roughly 3200 characters, around 700 "
                "tokens. That is the comfortable middle: enough evidence, far from the length "
                "where context starts diluting attention.",
            ],
            [
                "Context dilution: past a certain length, extra blocks reduce accuracy "
                "instead of improving it, and the effect is worst for facts buried in the "
                "middle of the prompt.",
                "Citation drift: models sometimes cite block 2 while quoting block 1. "
                "Rate-based citation metrics cannot catch this; claim-by-claim verification "
                "can.",
                "Instruction leakage: if the system prompt contains the evaluation rubric, the "
                "model starts writing answers that look like rubric items. Keep evaluation "
                "prompts out of the product prompt.",
            ],
        ],
    },
    "pdf-internals": {
        "title": "How a PDF File Actually Stores Content",
        "pages": [
            [
                "Text in a PDF is stored as character codes plus the font needed to draw "
                "them. A file may embed only the glyphs it uses, a font subset, which keeps the "
                "file small.",
                "Because the fonts travel with the file, the characters render even on a "
                "device that has no such font installed. A typical sample embeds DejaVu Sans "
                "and DejaVu Sans Bold as subsets.",
                "Selectable text is a property of the extractor, not of the file: if the "
                "characters are encoded, an extractor can pull them out; if the page is a scan, "
                "it cannot.",
            ],
            [
                "PDF has no table object. A table is text and lines placed at exact "
                "positions, which is why copying a table out of a PDF often loses its columns.",
                "Tagged PDFs add a structure layer that marks up rows and cells. Most "
                "generated files are not tagged, so extraction sees a flat stream of words.",
                "A vector chart is drawn with path operators, lines, rectangles and text, "
                "rather than stored as a picture. It stays sharp at any zoom and extracts as a "
                "handful of stray labels.",
            ],
            [
                "Photos are stored as image objects. A 480 by 240 JPEG is embedded as it is, "
                "with the DCTDecode filter.",
                "A link is an annotation: a clickable rectangle on the page with an action, "
                "sometimes a URI and sometimes a jump to another page.",
                "Document metadata such as title, author and keywords is optional and "
                "frequently empty. Never rely on it for citations; use the page index from the "
                "extractor.",
            ],
        ],
    },
}
