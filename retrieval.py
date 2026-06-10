"""Milestone 4 — Embedding, vector store, and retrieval.

Pipeline stage 3-4 of The Unofficial Guide (SF restaurant RAG):

    cleaned chunks --> embed (all-MiniLM-L6-v2) --> ChromaDB --> semantic search

Retrieval approach (from planning.md):
    - Embedding model: all-MiniLM-L6-v2 via sentence-transformers
    - Vector store: ChromaDB (persisted locally), cosine similarity
    - Top-k: 5 chunks per query

Chunks come from the ingestion pipeline in ingest.py, so cleaning + chunking
stay in one place. Each chunk is stored with its source filename (and URL when
known) as metadata, so retrieved context can be attributed back to its origin.

Run directly to build the index and run the planning.md test questions:

    python retrieval.py            # build (if needed) + demo queries
    python retrieval.py --rebuild  # force a fresh re-embed
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from ingest import Chunk, chunk_documents, clean_documents, load_documents

# --- Configuration -----------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # per planning.md Retrieval Approach
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"  # persisted index
COLLECTION_NAME = "sf_restaurants"
TOP_K = 5  # per planning.md

# Cosine distance above this means a weak match (smaller distance = closer).
# Per the rubric: scores above ~0.6-0.7 indicate the retrieval isn't confident.
WEAK_MATCH_DISTANCE = 0.7

# Reciprocal Rank Fusion constant for hybrid search. The paper's default is 60,
# but on a small corpus (156 chunks) that flattens rank differences so a strong
# single-method hit (e.g. a BM25 keyword match) gets washed out. A smaller value
# lets a top keyword result surface even when its semantic rank is poor.
RRF_K = 10

# Test questions from planning.md (Evaluation Plan), used by the demo.
TEST_QUESTIONS = [
    "What do people say about the food quality and atmosphere at Zushi Puzzle?",
    "What does the Michelin guide say about the best restaurant in SF for a celebration meal?",
    "Which restaurant is facing parking or transit challenges?",
    "What restaurants would you recommend for a group dinner that offers good value for money?",
    "Which SF restaurant is recommended for a first-time visitor who wants classic local food?",
]


@dataclass
class RetrievedChunk:
    """One search result: a chunk plus where it came from and how close it was."""

    text: str
    source: str
    chunk_index: int
    distance: float  # raw cosine distance from Chroma; LOWER = more relevant
    url: str | None = None
    # Populated by hybrid search so we can see why a chunk ranked where it did.
    semantic_rank: int | None = None  # 1-based rank by cosine distance
    keyword_rank: int | None = None  # 1-based rank by BM25 score
    rrf_score: float | None = None  # fused Reciprocal Rank Fusion score

    @property
    def is_weak(self) -> bool:
        """True if this match is likely too far to be relevant (see rubric)."""
        return self.distance > WEAK_MATCH_DISTANCE


# --- Embedding ---------------------------------------------------------------

# The model is heavy to load (~80 MB, downloaded on first use), so cache it.
_model: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    """Load the all-MiniLM-L6-v2 model once and reuse it."""
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts into normalized vectors (for cosine similarity)."""
    model = get_embedder()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,  # unit vectors -> cosine == dot product
        show_progress_bar=len(texts) > 50,
    )
    return vectors.tolist()


# --- Vector store -------------------------------------------------------------

def get_chunks() -> list[Chunk]:
    """Run the ingestion pipeline (load -> clean -> chunk) to produce chunks."""
    return chunk_documents(clean_documents(load_documents()))


def _get_client() -> chromadb.api.ClientAPI:
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def build_index(rebuild: bool = False) -> chromadb.api.models.Collection.Collection:
    """Embed all chunks and store them in ChromaDB.

    If the collection already holds the same number of chunks and ``rebuild`` is
    False, the existing index is reused (no re-embedding). Pass ``rebuild=True``
    after changing the cleaning/chunking logic to force a fresh build.
    """
    client = _get_client()
    chunks = get_chunks()

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # nothing to delete on first run

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # match our normalized embeddings
    )

    if not rebuild and collection.count() == len(chunks):
        print(f"Index already built: {collection.count()} chunks. Reusing it.")
        return collection

    # (Re)build from scratch so stale entries can't linger.
    if collection.count():
        client.delete_collection(COLLECTION_NAME)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    print(f"Embedding {len(chunks)} chunks with {EMBEDDING_MODEL} ...")
    ids = [f"{c.source}::{c.chunk_index}" for c in chunks]
    documents = [c.text for c in chunks]
    embeddings = embed_texts(documents)
    metadatas = [
        {
            "source": c.source,
            "chunk_index": c.chunk_index,
            "token_estimate": c.token_estimate,
        }
        for c in chunks
    ]

    collection.add(
        ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
    )
    print(f"Stored {collection.count()} chunks in {CHROMA_DIR}")
    return collection


# --- Retrieval ----------------------------------------------------------------

def retrieve(query: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
    """Return the ``top_k`` chunks most semantically similar to ``query``.

    Assumes the index has been built (call build_index() first, or run this
    module as a script). Embeds the query with the same model, then runs cosine
    nearest-neighbor search in ChromaDB.
    """
    client = _get_client()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    if collection.count() == 0:
        raise RuntimeError(
            "The vector store is empty. Run `python retrieval.py` to build it."
        )

    query_embedding = embed_texts([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved: list[RetrievedChunk] = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        retrieved.append(
            RetrievedChunk(
                text=doc,
                source=meta.get("source", "?"),
                chunk_index=int(meta.get("chunk_index", -1)),
                distance=dist,  # keep Chroma's raw cosine distance (lower = closer)
                url=meta.get("url"),
            )
        )
    return retrieved


# --- Hybrid retrieval (semantic + BM25 keyword) -------------------------------

# BM25 needs the chunk texts in memory; cache the index so we build it once.
_bm25: BM25Okapi | None = None
_bm25_chunks: list[Chunk] | None = None


def _tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens for BM25 (simple, language-agnostic)."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _get_bm25() -> tuple[BM25Okapi, list[Chunk]]:
    """Build (once) a BM25 index over the same chunks as the vector store."""
    global _bm25, _bm25_chunks
    if _bm25 is None:
        _bm25_chunks = get_chunks()
        _bm25 = BM25Okapi([_tokenize(c.text) for c in _bm25_chunks])
    return _bm25, _bm25_chunks


def retrieve_hybrid(
    query: str, top_k: int = TOP_K, rrf_k: int = RRF_K
) -> list[RetrievedChunk]:
    """Retrieve by fusing semantic (cosine) and keyword (BM25) rankings.

    Uses Reciprocal Rank Fusion (RRF): each chunk scores 1/(rrf_k + rank) from
    each ranker, summed. RRF works on ranks, not raw scores, so it sidesteps the
    fact that cosine distance and BM25 scores are on totally different scales.
    A chunk that ranks well in *either* method surfaces — which is exactly what
    rescues a rare keyword (e.g. "parking") buried in one diluted chunk.
    """
    # --- Semantic ranking over ALL chunks (so every chunk gets a rank + distance).
    client = _get_client()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    if collection.count() == 0:
        raise RuntimeError(
            "The vector store is empty. Run `python retrieval.py` to build it."
        )
    query_embedding = embed_texts([query])[0]
    sem = collection.query(
        query_embeddings=[query_embedding],
        n_results=collection.count(),
        include=["metadatas", "distances"],
    )
    sem_ids = sem["ids"][0]
    sem_dist = {cid: d for cid, d in zip(sem_ids, sem["distances"][0])}
    sem_rank = {cid: i + 1 for i, cid in enumerate(sem_ids)}  # already distance-sorted

    # --- Keyword (BM25) ranking over the same chunks.
    bm25, chunks = _get_bm25()
    scores = bm25.get_scores(_tokenize(query))
    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    kw_rank = {
        f"{chunks[i].source}::{chunks[i].chunk_index}": rank + 1
        for rank, i in enumerate(order)
    }
    id_to_chunk = {f"{c.source}::{c.chunk_index}": c for c in chunks}

    # --- Reciprocal Rank Fusion across both rankings.
    fused: dict[str, float] = {}
    for cid in id_to_chunk:
        s = 0.0
        if cid in sem_rank:
            s += 1.0 / (rrf_k + sem_rank[cid])
        if cid in kw_rank:
            s += 1.0 / (rrf_k + kw_rank[cid])
        fused[cid] = s

    top_ids = sorted(fused, key=lambda c: fused[c], reverse=True)[:top_k]

    results: list[RetrievedChunk] = []
    for cid in top_ids:
        c = id_to_chunk[cid]
        results.append(
            RetrievedChunk(
                text=c.text,
                source=c.source,
                chunk_index=c.chunk_index,
                distance=sem_dist.get(cid, 1.0),  # semantic distance for reference
                semantic_rank=sem_rank.get(cid),
                keyword_rank=kw_rank.get(cid),
                rrf_score=fused[cid],
            )
        )
    return results


# --- Driver -------------------------------------------------------------------

def _short(r: RetrievedChunk) -> str:
    return f"{r.source} #chunk{r.chunk_index}"


def _demo(query: str, top_k: int = TOP_K, hybrid: bool = False) -> None:
    print("\n" + "=" * 72)
    print(f"QUERY: {query}   ({'HYBRID' if hybrid else 'SEMANTIC'})")
    print("=" * 72)
    results = retrieve_hybrid(query, top_k) if hybrid else retrieve(query, top_k)
    for rank, r in enumerate(results, start=1):
        preview = r.text.replace("\n", " ").strip()
        if len(preview) > 240:
            preview = preview[:240] + " ..."
        flag = "  [WEAK]" if r.is_weak else ""
        extra = ""
        if hybrid:
            extra = f"  [sem #{r.semantic_rank}, kw #{r.keyword_rank}]"
        print(f"\n[{rank}] {_short(r)}  (distance {r.distance:.3f}){flag}{extra}")
        print(f"    {preview}")


def _compare(query: str, top_k: int = TOP_K) -> None:
    """Print semantic-only vs hybrid top-k side by side for one query."""
    sem = retrieve(query, top_k)
    hyb = retrieve_hybrid(query, top_k)
    print("\n" + "=" * 72)
    print(f"COMPARE: {query}")
    print("=" * 72)
    print(f"{'rank':<5}{'SEMANTIC-ONLY':<48}HYBRID (semantic + BM25)")
    print("-" * 72)
    for i in range(top_k):
        s = f"{_short(sem[i])} ({sem[i].distance:.2f})" if i < len(sem) else ""
        h = f"{_short(hyb[i])} ({hyb[i].distance:.2f})" if i < len(hyb) else ""
        print(f"{i + 1:<5}{s:<48}{h}")


def main() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Build index and run demo queries.")
    parser.add_argument(
        "--rebuild", action="store_true", help="Force a fresh re-embed of all chunks."
    )
    parser.add_argument("--query", help="Run a single query instead of the test set.")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--hybrid", action="store_true", help="Use hybrid (semantic + BM25) retrieval."
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show semantic-only vs hybrid results side by side.",
    )
    args = parser.parse_args()

    build_index(rebuild=args.rebuild)

    queries = [args.query] if args.query else TEST_QUESTIONS
    for q in queries:
        if args.compare:
            _compare(q, args.top_k)
        else:
            _demo(q, args.top_k, hybrid=args.hybrid)


if __name__ == "__main__":
    main()
