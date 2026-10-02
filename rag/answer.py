"""Grounded answer assembly with cite-or-refuse behaviour.

The model only ever sees numbered retrieved sources and is told to cite them; when
retrieval returns nothing we refuse without calling the LLM at all, because the
safest way to avoid an ungrounded answer is to never give the model the chance.
"""

from __future__ import annotations

from dataclasses import dataclass

import rag.llm
from rag.types import Retrieved

REFUSAL_MESSAGE = "I couldn't find anything in the indexed documents to answer that, so I won't guess."

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered sources provided. "
    "Cite the sources you use inline as [S1], [S2], etc. "
    "If the sources are insufficient to answer, say that you do not know "
    "instead of guessing. Never use outside knowledge."
)


@dataclass
class Answer:
    """A final answer: its text, the sources it was grounded on, and whether it refused."""

    text: str
    sources: list[Retrieved]
    refused: bool


def _build_prompt(question: str, retrieved: list[Retrieved]) -> str:
    """Format numbered, source-labelled context blocks followed by the question."""
    blocks = [f"[S{i}] ({r.chunk.source})\n{r.chunk.text}" for i, r in enumerate(retrieved, 1)]
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def answer_question(question: str, retrieved: list[Retrieved]) -> Answer:
    """Answer from retrieved sources, or refuse (without an LLM call) if there are none."""
    if not retrieved:
        return Answer(REFUSAL_MESSAGE, [], True)
    text = rag.llm.complete(_build_prompt(question, retrieved), system=SYSTEM_PROMPT)
    return Answer(text=text, sources=retrieved, refused=False)
