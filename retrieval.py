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
from dataclasses import dataclass
from pathlib import Path

import chromadb
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


# --- Driver -------------------------------------------------------------------

def _demo(query: str, top_k: int = TOP_K) -> None:
    print("\n" + "=" * 72)
    print(f"QUERY: {query}")
    print("=" * 72)
    for rank, r in enumerate(retrieve(query, top_k), start=1):
        preview = r.text.replace("\n", " ").strip()
        if len(preview) > 280:
            preview = preview[:280] + " ..."
        flag = "  [WEAK]" if r.is_weak else ""
        print(f"\n[{rank}] {r.source} #chunk{r.chunk_index}  (distance {r.distance:.3f}){flag}")
        print(f"    {preview}")


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
    args = parser.parse_args()

    build_index(rebuild=args.rebuild)

    if args.query:
        _demo(args.query, args.top_k)
    else:
        for question in TEST_QUESTIONS:
            _demo(question, args.top_k)


if __name__ == "__main__":
    main()
