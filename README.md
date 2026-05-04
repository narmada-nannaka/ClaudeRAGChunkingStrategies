# RAG Chunking Strategies

A practical proof-of-concept exploring the full pipeline of a Retrieval-Augmented Generation (RAG) system — from chunking and embedding to vector search, keyword search, and hybrid retrieval. Each notebook builds on the previous one, using `report.md` (a realistic multi-section markdown document) as the test corpus throughout.

---

## Project Structure

```
ClaudeRAGChunkingStrategies/
├── 001_chunking.ipynb    # Three chunking strategies (size, structure, semantic)
├── 002_embeddings.ipynb  # Embedding generation with VoyageAI
├── 003_vectordb.ipynb    # Custom vector index + semantic search
├── 004_bm25.ipynb        # Custom BM25 index + keyword search
├── 005_hybrid.ipynb      # Hybrid retriever using RRF over vector + BM25
├── report.md             # Sample test document (annual research review)
├── LICENSE               # MIT License
└── README.md
```

---

## What is Chunking?

In RAG systems, documents must be split into smaller pieces ("chunks") before being embedded and stored in a vector database. The chunking strategy directly affects retrieval quality: chunks that are too large dilute relevance signals; chunks that are too small lose context.

---

## Notebook 001 — Chunking Strategies

Implements three chunking approaches, all using only Python's `re` standard library module.

### 1. Size-based Chunking (`chunk_by_size`)

Splits text into fixed-size character windows with configurable overlap between adjacent chunks.

```python
chunk_by_size(text, chunk_size=150, chunk_overlap=20)
```

| Parameter | Description | Default |
|---|---|---|
| `chunk_size` | Max characters per chunk | 150 |
| `chunk_overlap` | Characters shared between consecutive chunks | 20 |

**Best for:** Uniform, unstructured text where sentence or section boundaries don't matter.  
**Drawback:** May split mid-word or mid-sentence, breaking semantic meaning.

---

### 2. Structure-based Chunking (`chunk_by_structure`)

Uses a regex to detect sentence boundaries (`.`, `!`, `?`) and groups a configurable number of sentences per chunk, with sentence-level overlap.

```python
chunk_by_structure(text, max_sentences_per_chunk=5, overlap_sentences=1)
```

| Parameter | Description | Default |
|---|---|---|
| `max_sentences_per_chunk` | Max sentences per chunk | 5 |
| `overlap_sentences` | Sentences shared with the next chunk | 1 |

**Best for:** Prose documents (articles, reports, transcripts) where the sentence is the meaningful unit of information.  
**Drawback:** Chunk sizes vary with sentence length; very long sentences can create oversized chunks.

---

### 3. Semantic-based Chunking (`chunk_by_semantic`)

Exploits the markdown structure of the input document. A regex splits on `## ` section headers, treating each section as a self-contained semantic unit.

```python
chunk_by_semantic(document_text)
```

No parameters — the document's own markdown structure drives the split boundaries.

**Best for:** Structured markdown documents (reports, wikis, documentation) where each section covers a distinct topic.  
**Drawback:** Chunk size is entirely determined by section length, which can vary dramatically.

---

### Choosing a Strategy

| Strategy | Splitting Mechanism | Document Type | Preserves Semantics |
|---|---|---|---|
| Size-based | Character count | Any | Low |
| Structure-based | Regex sentence boundaries | Prose / narrative | Medium |
| Semantic-based | Markdown section headers | Structured markdown | High |

---

## Notebook 002 — Embeddings

Generates dense vector embeddings for text chunks using the VoyageAI API (`voyage-3-large` model).

**Key function:**

```python
generate_embedding(text, model="voyage-3-large", input_type="query")
```

Chunks `report.md` by markdown section (semantic chunking), then generates an embedding vector for each chunk. The output is a high-dimensional float list suitable for similarity search.

**Requires:** `VOYAGE_API_KEY` in `.env`

---

## Notebook 003 — Vector Database

Implements a custom in-memory `VectorIndex` class and demonstrates a full end-to-end semantic search pipeline.

### `VectorIndex`

| Feature | Details |
|---|---|
| Distance metrics | Cosine distance, Euclidean distance |
| Storage | In-memory lists of vectors and document dicts |
| Documents | Each stored as `{"content": str, ...}` |

**Key methods:**

```python
store = VectorIndex(distance_metric="cosine", embedding_fn=generate_embedding)
store.add_vector(vector, document)   # Add a pre-computed vector
store.add_document(document)         # Embed and add in one call
store.search(query, k=2)             # Returns [(document, distance), ...]
```

### End-to-end pipeline

```
report.md → chunk_by_section → generate_embedding (VoyageAI) → VectorIndex → semantic search
```

**Example query:** `"What happened with INC-2023-Q4-011?"` retrieves the Cybersecurity and Software Engineering sections with the lowest cosine distances.

**Requires:** `VOYAGE_API_KEY` in `.env`

---

## Notebook 004 — BM25 Keyword Search

Implements a custom `BM25Index` class for lexical (keyword) retrieval — no embeddings or external APIs required.

### BM25 Algorithm

BM25 (Best Match 25) is a classical probabilistic ranking function that scores documents based on term frequency (TF) and inverse document frequency (IDF), with length normalization.

### `BM25Index`

| Parameter | Description | Default |
|---|---|---|
| `k1` | Term frequency saturation | 1.5 |
| `b` | Document length normalization | 0.75 |
| `tokenizer` | Optional custom tokenizer | lowercase + `\W+` split |

**Key methods:**

```python
store = BM25Index(k1=1.5, b=0.75)
store.add_document({"content": chunk})   # Add a document
store.search("query text", k=3)          # Returns [(document, normalized_score), ...]
```

Scores are normalized using `exp(-factor * raw_score)` so lower values indicate higher relevance (consistent with distance convention used in `VectorIndex`).

**Example query:** `"What happened with INC-2023-Q4-011?"` correctly surfaces the Software Engineering and Cybersecurity sections via exact keyword matching.

---

## Notebook 005 — Hybrid Search (RRF)

Combines `VectorIndex` (semantic) and `BM25Index` (lexical) into a unified `Retriever` using **Reciprocal Rank Fusion (RRF)** to merge ranked result lists from both indexes.

### `Retriever`

Accepts any number of indexes that implement the `SearchIndex` protocol:

```python
vector_index = VectorIndex(embedding_fn=generate_embedding)
bm25_index   = BM25Index()

retriever = Retriever(bm25_index, vector_index)
retriever.add_documents([{"content": chunk} for chunk in chunks])
results = retriever.search("query text", k=3)
```

### Reciprocal Rank Fusion

For each candidate document, RRF sums `1 / (k_rrf + rank)` across all indexes (default `k_rrf=60`). This rank-based fusion is robust to score scale differences between BM25 and cosine distance and does not require re-scaling.

### End-to-end pipeline

```
report.md → chunk_by_section → Retriever.add_documents
                                  ├── VectorIndex (embed + cosine similarity)
                                  └── BM25Index (tokenize + TF-IDF scoring)
                                       ↓
                                  RRF fusion → top-k results
```

**Requires:** `VOYAGE_API_KEY` in `.env`

---

## Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

| Notebook | External Dependencies |
|---|---|
| `001_chunking.ipynb` | None (standard library only) |
| `002_embeddings.ipynb` | `voyageai`, `python-dotenv` |
| `003_vectordb.ipynb` | `voyageai`, `python-dotenv` |
| `004_bm25.ipynb` | None (standard library only) |
| `005_hybrid.ipynb` | `voyageai`, `python-dotenv` |

---

## Setup

1. Clone the repository:

   ```bash
   git clone <repo-url>
   cd ClaudeRAGChunkingStrategies
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate

   # macOS/Linux
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install voyageai python-dotenv
   ```

4. Create a `.env` file in the project root with your API keys:

   ```
   VOYAGE_API_KEY=your_voyage_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

5. Launch Jupyter:

   ```bash
   jupyter notebook
   ```

6. Open the notebooks in order: `001_chunking.ipynb` → `005_hybrid.ipynb`.

---

## License

MIT — see [LICENSE](LICENSE) for details.
