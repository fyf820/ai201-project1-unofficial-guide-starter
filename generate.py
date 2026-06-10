"""Milestone 5 — Grounded generation and query interface.

Final pipeline stage of The Unofficial Guide (SF restaurant RAG):

    query --> retrieve top-k chunks --> ground LLM on that context --> answer + sources

Generation:
    - LLM: Groq (chat completions), model from GROQ_MODEL (default llama-3.3-70b).
    - Grounding: the model is instructed to answer ONLY from the retrieved
      context and to refuse when the context doesn't cover the question. We also
      filter out low-relevance chunks (cosine distance above WEAK_MATCH_DISTANCE)
      before they ever reach the prompt, so weak retrievals can't be fabricated
      into an answer.
    - Output: a written answer followed by a Sources list (the documents the
      answer was actually drawn from), for attribution.

Interface (Gradio web UI):
    python generate.py --app          # launch the web UI
    python generate.py --query "..."  # one-shot answer in the terminal
    python generate.py                # run the planning.md test questions

Requires GROQ_API_KEY in a local .env file (see .env.example).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field

from dotenv import load_dotenv
from groq import Groq

from retrieval import (
    TOP_K,
    WEAK_MATCH_DISTANCE,
    RetrievedChunk,
    TEST_QUESTIONS,
    build_index,
    retrieve,
    retrieve_hybrid,
)

load_dotenv()

# Groq model id — override with GROQ_MODEL in .env if you like.
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# The grounding instruction. This is the core of "answers from context only".
SYSTEM_PROMPT = """You are a knowledgeable guide to San Francisco restaurants. \
Answer the question using only the information in the provided documents.

Rules:
1. Use ONLY information found in the CONTEXT documents. Do not rely on outside knowledge.
2. If the documents don't contain enough information to answer, say exactly: \
"I don't have enough information on that."
3. Name specific restaurants and quote concrete details (dishes, prices, \
atmosphere, wait times) when they appear in the context.
4. After each claim, cite the source filename(s) it came from in brackets, \
e.g. [8-yelp-zushi-puzzle.txt].
5. Be concise and do not invent restaurants, reviews, or facts."""

# When every retrieved chunk is too far to trust, we don't even call the LLM.
REFUSAL = "I don't have enough information on that."


@dataclass
class AnswerResult:
    """A grounded answer plus the sources it was drawn from."""

    query: str
    answer: str
    sources: list[dict] = field(default_factory=list)  # {source, url, distance}

    def to_markdown(self) -> str:
        lines = [self.answer.strip(), ""]
        if self.sources:
            lines.append("**Sources:**")
            for s in self.sources:
                label = s["source"]
                if s.get("url"):
                    label = f"[{s['source']}]({s['url']})"
                lines.append(f"- {label}  _(distance {s['distance']:.3f})_")
        return "\n".join(lines)

    def to_text(self) -> str:
        out = [self.answer.strip()]
        if self.sources:
            out.append("\nSources:")
            out += [f"  - {s['source']} (distance {s['distance']:.3f})" for s in self.sources]
        return "\n".join(out)


# --- Prompt assembly ----------------------------------------------------------

def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as numbered, source-labeled context blocks."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] (source: {c.source})\n{c.text}")
    return "\n\n".join(blocks)


def _unique_sources(chunks: list[RetrievedChunk]) -> list[dict]:
    """De-duplicate sources, keeping the best (smallest) distance per document."""
    best: dict[str, dict] = {}
    for c in chunks:
        if c.source not in best or c.distance < best[c.source]["distance"]:
            best[c.source] = {"source": c.source, "url": c.url, "distance": c.distance}
    return sorted(best.values(), key=lambda s: s["distance"])


# --- Generation ---------------------------------------------------------------

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key or key == "your_key_here":
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = Groq(api_key=key)
    return _client


def answer_question(query: str, top_k: int = TOP_K, hybrid: bool = True) -> AnswerResult:
    """Retrieve context, then generate a grounded answer with source attribution.

    Hybrid retrieval (semantic + BM25) is the default because it surfaces chunks
    that keyword-match a rare term even when their semantic rank is poor (e.g. the
    parking note that pure semantic search buried). Pass hybrid=False for
    semantic-only.
    """
    retrieved = retrieve_hybrid(query, top_k) if hybrid else retrieve(query, top_k)

    # Grounding guard: drop low-relevance chunks so they can't seed a hallucination.
    strong = [c for c in retrieved if not c.is_weak]
    if not strong:
        return AnswerResult(query=query, answer=REFUSAL, sources=[])

    context = _format_context(strong)
    user_prompt = (
        f"CONTEXT (the only information you may use):\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer the question using only the information in the documents above. "
        "If they don't contain enough information, say \"I don't have enough "
        "information on that.\" Cite the source filename(s) you used."
    )

    client = _get_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,  # low temperature keeps it close to the source text
    )
    answer = response.choices[0].message.content

    return AnswerResult(query=query, answer=answer, sources=_unique_sources(strong))


# --- Gradio interface ---------------------------------------------------------

def launch_app() -> None:
    """Launch a simple Gradio web UI for asking questions."""
    import gradio as gr

    def respond(question: str) -> str:
        question = (question or "").strip()
        if not question:
            return "Please enter a question."
        return answer_question(question).to_markdown()

    with gr.Blocks(title="The Unofficial Guide — SF Restaurants") as demo:
        gr.Markdown(
            "# 🍽️ The Unofficial Guide — San Francisco Restaurants\n"
            "Ask about SF restaurants. Answers come **only** from the collected "
            "reviews and guides, with sources listed."
        )
        question = gr.Textbox(
            label="Your question",
            placeholder="e.g. What do people say about Zushi Puzzle?",
            lines=2,
        )
        ask = gr.Button("Ask", variant="primary")
        answer = gr.Markdown(label="Answer")

        ask.click(fn=respond, inputs=question, outputs=answer)
        question.submit(fn=respond, inputs=question, outputs=answer)

        gr.Examples(examples=[[q] for q in TEST_QUESTIONS], inputs=question)

    demo.launch()


# --- Driver -------------------------------------------------------------------

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Grounded RAG answers over SF restaurants.")
    parser.add_argument("--app", action="store_true", help="Launch the Gradio web UI.")
    parser.add_argument("--query", help="Answer a single question in the terminal.")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Use semantic-only retrieval (default is hybrid semantic + BM25).",
    )
    args = parser.parse_args()
    hybrid = not args.semantic

    build_index()  # ensure the vector store exists before we answer anything

    if args.app:
        launch_app()
        return

    queries = [args.query] if args.query else TEST_QUESTIONS
    for q in queries:
        print("\n" + "=" * 72)
        print(f"Q: {q}   ({'hybrid' if hybrid else 'semantic'} retrieval)")
        print("=" * 72)
        print(answer_question(q, args.top_k, hybrid=hybrid).to_text())


if __name__ == "__main__":
    main()
