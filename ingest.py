"""Milestone 3 — Document ingestion, cleaning, and chunking.

Pipeline stage 1-2 of The Unofficial Guide (San Francisco restaurant RAG):

    documents/*.txt  -->  load  -->  clean  -->  paragraph chunks

Chunking strategy (from planning.md):
    - Paragraph-based chunks, ~300-400 tokens each
    - 20-50 token overlap between consecutive chunks
    - Rationale: sources are short reviews / single-paragraph-per-restaurant
      writeups, so paragraph boundaries keep each restaurant's meaning intact.

Token counts here are an estimate (words + standalone punctuation), not the
exact WordPiece count the embedding model uses. That's fine for sizing chunks
and keeps this script dependency-free; the real tokenizer runs at embed time.

Run directly to load, clean, chunk, and print samples for inspection:

    python ingest.py
"""

from __future__ import annotations

import html
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --- Configuration -----------------------------------------------------------

# Corpus folder, resolved relative to this file so the script runs from anywhere.
DOCUMENTS_DIR = Path(__file__).resolve().parent / "documents"

# Plain-text formats we read directly. (.pdf handled separately if needed.)
TEXT_SUFFIXES = {".txt", ".md"}

# Files inside documents/ that are not corpus content.
SKIP_NAMES = {".gitkeep", "source_fetch_notes.txt"}

# Chunk sizing in estimated tokens (see module docstring).
TARGET_CHUNK_TOKENS = 350  # aim for the middle of the 300-400 range
MAX_CHUNK_TOKENS = 400  # hard ceiling before we force a split
OVERLAP_TOKENS = 30  # carried from the tail of one chunk into the next (20-50)


# --- Data model --------------------------------------------------------------

@dataclass
class Document:
    """A single source document after loading and cleaning."""

    source: str  # file name, e.g. "1-eater-sf-38-best-restaurants.txt"
    text: str  # cleaned full text
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrievable slice of a document."""

    text: str
    source: str  # which document it came from
    chunk_index: int  # position within that document (0-based)
    token_estimate: int


# --- Stage 1: load -----------------------------------------------------------

def load_documents(documents_dir: Path | str = DOCUMENTS_DIR) -> list[Document]:
    """Load every corpus document, returning raw (uncleaned) text + metadata."""
    documents_dir = Path(documents_dir)
    if not documents_dir.is_dir():
        raise FileNotFoundError(f"Documents folder not found: {documents_dir}")

    docs: list[Document] = []
    for path in sorted(documents_dir.iterdir()):
        if not path.is_file() or path.name in SKIP_NAMES:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            print(f"Skipping unsupported file type: {path.name}")
            continue

        raw = path.read_text(encoding="utf-8", errors="replace")
        if not raw.strip():
            print(f"Skipping empty file: {path.name}")
            continue

        docs.append(Document(source=path.name, text=raw, metadata={"source": path.name}))

    return docs


# --- Stage 2: clean ----------------------------------------------------------

# A "Source:/URL:/Extracted text:" header sits atop the scraped files. We pull
# the URL out as metadata, then drop the header from the body.
_URL_RE = re.compile(r"^URL:\s*(\S+)", re.MULTILINE)
_HEADER_RE = re.compile(
    r"\A(?:Source:.*\n|URL:.*\n|Extracted text:\s*\n|\s*\n)+", re.IGNORECASE
)

# "(image: ...)" placeholders left by the scraper carry no text content.
_IMAGE_PLACEHOLDER_RE = re.compile(r"\(image:[^)]*\)", re.IGNORECASE)

# Any leftover HTML tags.
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# UI boilerplate (Reddit / Quora / Eater / SF Chronicle) that appears as
# standalone lines and carries no review content.
_BOILERPLATE_LINE_RE = re.compile(
    r"""^\s*(
        •                                   # bullet separator
        | \d+\s*(?:y|yr|yrs|mo|d|h|hr|hrs|min|w)(?:\s*ago)?   # "1y ago", bare "5y", "6mo"
        | \d+\s+(?:upvotes?|downvotes?|comments?|points?|replies?|likes?|views?)
        | (?:Reply|Share|Save|Award|Report|Follow|Upvote|Downvote|Edited)
        | Continue\ this\ thread
        | View\ \d+\ (?:more\ )?(?:replies|comments)
        | Profile\ photo\ for\ .*
        | u/\S+\ avatar                      # Reddit avatar alt-text "u/name avatar"
        | \[deleted\] | \[removed\]          # deleted/removed comment placeholders
        | \d+\ more\ repl(?:y|ies)           # "1 more reply", "10 more replies"
        | Promoted                           # promoted/ad marker
        # --- site chrome / calls-to-action ---
        | Read\ more\s*\+?
        | See\ more
        | Visit\ website
        | Website                            # bare link label before an address
        | .*_(?:credit|sized).*              # image alt-text filenames, "X_credit..._sized"
        | Book\ a\ table.*
        | Have\ you\ been\??
        | Wish\ list
        | Yes | No                          # SF Chronicle "Have you been?" toggle
        | Address                            # bare field label (street kept below)
        | Current\ eater\ city:.*
        | Originally\ Answered:.*
        # --- author bylines & photo credits ---
        | (?:Lives?|Lived)\ (?:in|here)\b.*  # "Lives in SF...", "Lived here for 4 years..."
        | .*Author\ has\ .*answers?.*
        | .*\d{1,2}y                          # Quora byline age suffix, e.g. "...settling here.10y"
        | \(?\+?\d[\d()\s\-]{7,}\d\)?        # phone numbers
        | .*\ /\ S\.F\.\ Chronicle           # "Jess Lander / S.F. Chronicle"
        | [A-Z][\w.\ ]+\ Photography          # "Michelle Chou Photography"
        # --- Yelp profile chrome ---
        | \d+                                # bare counter lines (friends/photos)
        | \d+\ photos?(?:\d+\ check-?in)?    # "6 photos", "10 photos1 check-in"
        | \d+\ check-?ins?
        | Elite\ \d+
        | \d+\ \w+\ reviews?                 # "4 Sushi reviews", "15 Salad reviews"
        | See\ all\ photos\ from\ .*
        # --- SF Chronicle site chrome (article title & byline kept) ---
        | San\ Francisco\ Chronicle\ logo
        | Advertisement
        | (?:The\ Top\ 100\ Restaurants\ is\ )?[Ss]upported\ by.*
        | OpenTable\ and\ Visa.*
        | My\ Stats | Filters | List | Map   # masthead nav widgets
        | Published\d.*                      # "Published2025"
        | Section\w+                         # "SectionFood"
        | Price\$.*                          # "Price$—$$$$" header label
        | Track\ how\ many\ restaurants.*
        | Top\ .{2,45}(?:restaurants?|spots?|fine\ dining|in\ the\ Bay\ Area)  # related-article links
        | Reserve\ on | OpenTable            # reservation widget
    )\s*$""",
    re.IGNORECASE | re.VERBOSE,
)

# Standalone date lines that trail related-article cards ("Jan 27, 2025", "Mar 19").
# Only stripped when they directly follow a duplicated card (see below), so
# genuine review dates elsewhere are preserved.
_DATE_LINE_RE = re.compile(
    r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)\.?\s+\d{1,2}(?:,\s*\d{4})?$",
    re.IGNORECASE,
)

# A line that is one token repeated with no separator, e.g. "SpruceSpruce" — a
# scrape artifact where two identical image captions merged into one line.
_SELF_DOUBLED_RE = re.compile(r"^(\S{4,})\1$")


def _collapse_link_cards(lines: list[str]) -> list[str]:
    """Remove repeated link cards / image-caption echoes.

    Two patterns in this corpus repeat a line as noise:
      - Eater related-article cards: the same headline twice, often + a date.
      - jsfashionista image captions: the restaurant name echoed several times
        after its write-up, separated by blank lines (e.g. "Zuni Café" x4).
    We treat blank lines as transparent, so any line that recurs 2+ times with
    only blanks between is dropped entirely (all copies), along with a trailing
    date line. Genuine prose is never repeated this way, so content is safe.
    """
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        current = lines[i].strip()
        if not current:
            out.append(lines[i])
            i += 1
            continue
        # Scan forward for identical lines, skipping blank lines between.
        last_match, count, k = i, 1, i + 1
        while k < n:
            s = lines[k].strip()
            if s == "":
                k += 1
                continue
            if s == current:
                count += 1
                last_match = k
                k += 1
                continue
            break
        if count >= 2:  # repeated line -> drop all copies (and blanks between)
            i = last_match + 1
            while i < n and lines[i].strip() == "":
                i += 1
            if i < n and _DATE_LINE_RE.match(lines[i].strip()):
                i += 1  # drop the card's trailing date too
            continue
        out.append(lines[i])
        i += 1
    return out


def clean_text(text: str) -> tuple[str, dict]:
    """Clean one document. Returns (cleaned_text, extracted_metadata)."""
    metadata: dict = {}

    # Capture the source URL before we strip the header.
    url_match = _URL_RE.search(text)
    if url_match:
        metadata["url"] = url_match.group(1)

    # Drop the leading Source/URL/Extracted-text header block.
    text = _HEADER_RE.sub("", text, count=1)

    # Decode HTML entities (&amp; -> &, &nbsp; -> space) and strip tags.
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub("", text)

    # Remove scraper image placeholders.
    text = _IMAGE_PLACEHOLDER_RE.sub("", text)

    # Drop boilerplate UI lines and normalize whitespace line by line.
    kept_lines: list[str] = []
    for line in text.splitlines():
        line = line.replace(" ", " ").rstrip()  # nbsp -> space
        if _BOILERPLATE_LINE_RE.match(line):
            continue
        if _SELF_DOUBLED_RE.match(line.strip()):  # "SpruceSpruce" merged caption
            continue
        kept_lines.append(line)

    # Remove duplicated related-article link cards (and their trailing dates).
    kept_lines = _collapse_link_cards(kept_lines)

    text = "\n".join(kept_lines)

    # Collapse 3+ blank lines into a single blank line (paragraph separator).
    text = re.sub(r"\n[ \t]*\n[ \t]*(\n[ \t]*)+", "\n\n", text)

    return text.strip(), metadata


def clean_documents(docs: list[Document]) -> list[Document]:
    """Apply clean_text to every document in place-ish (returns new list)."""
    cleaned: list[Document] = []
    for doc in docs:
        text, meta = clean_text(doc.text)
        merged = {**doc.metadata, **meta}
        cleaned.append(Document(source=doc.source, text=text, metadata=merged))
    return cleaned


# --- Stage 3: chunk ----------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Estimate token count as words + standalone punctuation marks.

    Slightly overcounts vs. whitespace words, landing closer to subword
    tokenizer output than a plain word count would.
    """
    return len(re.findall(r"\w+|[^\w\s]", text))


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines into paragraphs, dropping empties."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _split_sentences(text: str) -> list[str]:
    """Lightweight sentence split for paragraphs that exceed the max size."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _overlap_tail(text: str, overlap_tokens: int) -> str:
    """Return the last ~overlap_tokens words of text, for chunk overlap."""
    words = text.split()
    if len(words) <= overlap_tokens:
        return text
    return " ".join(words[-overlap_tokens:])


def chunk_text(
    text: str,
    target_tokens: int = TARGET_CHUNK_TOKENS,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[str]:
    """Paragraph-based chunking with token-sized overlap.

    Greedily packs whole paragraphs into a chunk until adding the next one
    would exceed ``target_tokens``. Paragraphs longer than ``max_tokens`` are
    split on sentence boundaries first so a single huge paragraph can't blow
    past the ceiling. Each chunk after the first is prefixed with the tail of
    the previous one to preserve cross-boundary context.
    """
    # Break oversized paragraphs into sentence-sized units up front.
    units: list[str] = []
    for para in _split_paragraphs(text):
        if estimate_tokens(para) <= max_tokens:
            units.append(para)
            continue
        sentence_buf = ""
        for sentence in _split_sentences(para):
            candidate = f"{sentence_buf} {sentence}".strip()
            if estimate_tokens(candidate) > max_tokens and sentence_buf:
                units.append(sentence_buf)
                sentence_buf = sentence
            else:
                sentence_buf = candidate
        if sentence_buf:
            units.append(sentence_buf)

    # Last resort: any unit still over the ceiling (e.g. a long list with no
    # sentence punctuation) gets hard-split on word boundaries.
    capped_units: list[str] = []
    for unit in units:
        if estimate_tokens(unit) <= max_tokens:
            capped_units.append(unit)
            continue
        buf: list[str] = []
        for word in unit.split():
            buf.append(word)
            if estimate_tokens(" ".join(buf)) >= max_tokens:
                capped_units.append(" ".join(buf))
                buf = []
        if buf:
            capped_units.append(" ".join(buf))
    units = capped_units

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}".strip() if current else unit
        if current and estimate_tokens(candidate) > target_tokens:
            chunks.append(current)
            overlap = _overlap_tail(current, overlap_tokens)
            current = f"{overlap}\n\n{unit}".strip()
        else:
            current = candidate
    if current.strip():
        chunks.append(current)

    return chunks


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    """Chunk every cleaned document into retrievable Chunk objects."""
    all_chunks: list[Chunk] = []
    for doc in docs:
        for i, piece in enumerate(chunk_text(doc.text)):
            all_chunks.append(
                Chunk(
                    text=piece,
                    source=doc.source,
                    chunk_index=i,
                    token_estimate=estimate_tokens(piece),
                )
            )
    return all_chunks


# --- Driver ------------------------------------------------------------------

def main() -> None:
    # Windows consoles default to cp1252 and choke on é, ō, — etc. Force UTF-8.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    raw_docs = load_documents()
    docs = clean_documents(raw_docs)
    chunks = chunk_documents(docs)

    print(f"Loaded and cleaned {len(docs)} documents from {DOCUMENTS_DIR}\n")
    for doc in docs:
        n = sum(1 for c in chunks if c.source == doc.source)
        print(f"  {doc.source:<48} {len(doc.text):>8,} chars  ->  {n:>3} chunks")
    print(f"\nTotal: {len(chunks)} chunks across {len(docs)} documents")

    sizes = [c.token_estimate for c in chunks]
    if sizes:
        print(
            f"Chunk size (est. tokens): min={min(sizes)} "
            f"max={max(sizes)} avg={sum(sizes) // len(sizes)}"
        )

    # --- Inspect one full cleaned document ---
    print("\n" + "=" * 70)
    print(f"SAMPLE CLEANED DOCUMENT: {docs[0].source}")
    print(f"metadata: {docs[0].metadata}")
    print("=" * 70)
    print(docs[0].text[:1500])
    print("... [truncated] ..." if len(docs[0].text) > 1500 else "")

    # --- Inspect 5 representative chunks, spread across the corpus ---
    print("\n" + "=" * 70)
    print("5 SAMPLE CHUNKS")
    print("=" * 70)
    step = max(1, len(chunks) // 5)
    for c in chunks[::step][:5]:
        print(f"\n--- {c.source} #chunk{c.chunk_index} (~{c.token_estimate} tokens) ---")
        print(c.text)


if __name__ == "__main__":
    main()
