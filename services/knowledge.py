"""
The knowledge base: every markdown file in knowledge/, concatenated.

WHY THERE IS NO VECTOR DATABASE HERE
------------------------------------
The obvious way to build this is RAG — chunk the documents, embed them, and
retrieve the few chunks nearest the question. That is the right answer for
hundreds of documents. For a knowledge base of a few pages it is strictly worse:

  - Retrieval can MISS. If the answer to "do you take apart beds?" sits in a
    paragraph that didn't score in the top 4, the agent answers "I don't have
    that" while the fact is sitting right there in the file. Handing over the
    whole thing cannot miss.
  - It adds an index that has to be rebuilt every time someone edits a sentence,
    which is exactly the sort of step that gets forgotten until the agent is
    quoting last quarter's rates.
  - It costs an embedding call per turn to save tokens we are not short of.

Everything the rest of the app knows about the knowledge base goes through
`all_context()`. If the material ever grows past what fits comfortably in a
prompt, swapping in retrieval means changing this one function and nothing else.

CACHING
-------
Files are read once and held in memory, keyed by the directory's newest mtime.
Editing a file and asking another question picks the change up immediately in
development, and production never touches the disk twice.
"""

import logging
from functools import lru_cache
from pathlib import Path

from services import config

logger = logging.getLogger(__name__)


class EmptyKnowledgeBase(RuntimeError):
    """
    Raised when knowledge/ has no readable content.

    Deliberately fatal rather than degrading to "answer from general knowledge".
    An agent with no reference material is an agent that invents rates, and it
    would do so quietly — the answers still read fine. Better to fail loudly at
    startup than to be wrong convincingly at 9pm on a Friday.
    """


def _fingerprint(directory: Path) -> tuple:
    """Cheap change-detector: the name and mtime of every markdown file."""
    if not directory.is_dir():
        return ()
    return tuple(
        sorted((p.name, p.stat().st_mtime_ns) for p in directory.glob("*.md"))
    )


@lru_cache(maxsize=4)
def _load(directory_str: str, _fp: tuple) -> str:
    directory = Path(directory_str)
    parts: list[str] = []

    for path in sorted(directory.glob("*.md")):
        # README.md documents the format for whoever edits the folder. It is
        # instructions to a human, not facts about the business, and feeding it
        # to the model just adds noise.
        if path.name.lower() == "readme.md":
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        parts.append(f"## Source: {path.name}\n\n{text}")

    if not parts:
        raise EmptyKnowledgeBase(
            f"No usable markdown files in {directory}. The agent has nothing to "
            "answer from — add at least one .md file before starting it."
        )

    logger.info("Knowledge base: %d file(s) from %s", len(parts), directory)
    return "\n\n---\n\n".join(parts)


def all_context() -> str:
    """
    The entire knowledge base as one string, ready to drop into a prompt.

    Raises EmptyKnowledgeBase if there is nothing to answer from.
    """
    directory = config.knowledge_dir()
    return _load(str(directory), _fingerprint(directory))


def describe() -> str:
    """One-line summary for startup logs and the status endpoint."""
    directory = config.knowledge_dir()
    try:
        text = all_context()
    except EmptyKnowledgeBase as exc:
        return f"knowledge: EMPTY ({exc})"
    files = len([p for p in directory.glob("*.md") if p.name.lower() != "readme.md"])
    # ~4 characters per token is close enough for a log line.
    return f"knowledge: {files} file(s), ~{len(text) // 4:,} tokens"
