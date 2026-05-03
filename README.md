# RAG Chunking Strategies

A practical proof-of-concept exploring three core text chunking strategies used in Retrieval-Augmented Generation (RAG) pipelines. The notebook demonstrates how different chunking approaches produce different results on the same document — helping you choose the right strategy for your use case.

---

## What is Chunking?

In RAG systems, documents must be split into smaller pieces ("chunks") before being embedded and stored in a vector database. The chunking strategy directly affects retrieval quality: chunks that are too large dilute relevance signals; chunks that are too small lose context. The three strategies here progress from simple size-awareness to deep semantic understanding.

---

## Strategies Implemented

### 1. Size-based Chunking (`chunk_by_size`)

Uses the total character count of the file to determine where chunk boundaries fall. Splits the text into fixed-size windows with configurable overlap between adjacent chunks.

```python
chunk_by_size(text, chunk_size=150, chunk_overlap=20)
```

| Parameter | Description | Default |
|---|---|---|
| `chunk_size` | Max characters per chunk | 150 |
| `chunk_overlap` | Characters shared between consecutive chunks | 20 |

**Best for:** Uniform, unstructured text where sentence or section boundaries don't matter. Fast and predictable output size.  
**Drawback:** May split mid-word or mid-sentence, breaking semantic meaning.

---

### 2. Structure-based Chunking (`chunk_by_structure`)

Uses a regular expression to detect sentence boundaries (`.`, `!`, `?`) and groups a configurable number of sentences per chunk, with sentence-level overlap between chunks.

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

The input document is written in markdown, so markdown semantics are exploited directly. A regex splits the document wherever a `## ` section header begins, treating each section as a self-contained semantic unit.

```python
chunk_by_semantic(document_text)
```

No parameters — the document's own markdown structure drives the split boundaries.

**Best for:** Structured markdown documents (reports, wikis, documentation) where each section covers a distinct topic.  
**Drawback:** Chunk size is entirely determined by section length, which can vary dramatically. No overlap between sections.

---

## Project Structure

```
ClaudeRAGChunkingStrategies/
├── 001_chunking.ipynb   # Main notebook with all three implementations
├── report.md            # Sample test document (annual research review)
├── LICENSE              # MIT License
└── README.md
```

**`report.md`** is a realistic multi-section markdown document covering 10 research domains. It is used as the test input for all three chunking strategies and is a good stand-in for enterprise reports, structured documentation, or knowledge-base articles.

---

## Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

All chunking implementations rely only on Python's standard library (`re` module) — no external packages required.

---

## Setup

1. Clone the repository:

   ```bash
   git clone <repo-url>
   cd ClaudeRAGChunkingStrategies
   ```

2. (Recommended) Create and activate a virtual environment:

   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate

   # macOS/Linux
   source .venv/bin/activate
   ```

3. Launch Jupyter:

   ```bash
   jupyter notebook
   ```

4. Open `001_chunking.ipynb`.

---

## Running the Notebook

The final cell loads `report.md` and applies a chunking strategy. Switch between strategies by uncommenting the relevant line:

```python
with open("./report.md", "r") as f:
    text = f.read()

# Uncomment the strategy you want to test:
# chunks = chunk_by_size(text, 500, 150)
# chunks = chunk_by_structure(text)
chunks = chunk_by_semantic(text)

[print(chunk + "\n----\n") for chunk in chunks]
```

Each chunk is printed separated by `----` so you can visually compare what each strategy produces.

---

## Choosing a Strategy

| Strategy | Splitting Mechanism | Document Type | Preserves Semantics |
|---|---|---|---|
| Size-based | Character count | Any | Low |
| Structure-based | Regex sentence boundaries | Prose / narrative | Medium |
| Semantic-based | Markdown section headers | Structured markdown | High |

A common production pattern is to combine strategies — for example, semantic chunking first (to respect document structure), then size-based chunking within large sections (to cap chunk size).

---

## License

MIT — see [LICENSE](LICENSE) for details.
