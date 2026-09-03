"""
Per-node model registry.

The ONLY module that knows a model provider exists. Nodes call `get_model("guard")`
and never import a chat class, so moving the whole agent from OpenAI to OpenRouter
is one environment variable rather than a refactor.

The nodes want genuinely different things. Deciding whether a message is about
moving is not the same job as writing the answer a customer reads, and paying for
the second model to do the first job on every single turn is how a public chat
agent gets expensive.
"""

import os
from dataclasses import dataclass

from langchain.chat_models import init_chat_model

from services import config


@dataclass(frozen=True)
class ModelSpec:
    """One node's model on each backend, plus why it was chosen."""

    openai: str
    openrouter: str
    rationale: str


REGISTRY: dict[str, ModelSpec] = {
    # Runs FIRST on every turn, on a public endpoint, so it is the thing most
    # exposed to junk traffic. It answers one narrowly-scoped structured
    # question, which the cheapest capable model does reliably.
    "guard": ModelSpec(
        openai="gpt-4.1-mini",
        openrouter="anthropic/claude-haiku-4.5",
        rationale="cheap binary safety classification on every turn",
    ),
    # Four-way lane classification. Also every turn, also cheap.
    "router": ModelSpec(
        openai="gpt-4.1-mini",
        openrouter="anthropic/claude-haiku-4.5",
        rationale="cheap 4-way routing on every turn",
    ),
    # Writes what the customer actually reads, constrained to the knowledge
    # base. The stakes are voice and grounding — staying inside the reference
    # material and never inventing a number — which is worth the better model.
    "knowledge": ModelSpec(
        openai="gpt-4.1",
        openrouter="anthropic/claude-sonnet-4.6",
        rationale="customer-visible prose, must not drift off the reference material",
    ),
    # Reads the conversation once at the start of the estimate flow and pulls
    # out what the customer already said. Narrow, structured, and everything it
    # returns is re-validated in Python, so cheap is fine.
    "extract": ModelSpec(
        openai="gpt-4.1-mini",
        openrouter="anthropic/claude-haiku-4.5",
        rationale="one structured read of the conversation, output is Python-validated",
    ),
    # Short in-persona lines: the refusal, the handoff, the form intro.
    "reply": ModelSpec(
        openai="gpt-4.1-mini",
        openrouter="anthropic/claude-haiku-4.5",
        rationale="short scripted-ish replies, low stakes",
    ),
}


def _model_name(node: str) -> str:
    """Resolve a node's model, honouring a per-node env override."""
    if node not in REGISTRY:
        raise KeyError(f"No model registered for node {node!r}. Known: {sorted(REGISTRY)}")

    # e.g. SUPPORT_MODEL_GUARD=gpt-4o-mini
    if override := os.getenv(f"SUPPORT_MODEL_{node.upper()}"):
        return override

    spec = REGISTRY[node]
    return spec.openrouter if config.model_backend() == "openrouter" else spec.openai


def _supports_temperature(model_name: str) -> bool:
    """
    The GPT-5 family rejects any temperature other than the default — passing
    temperature=0 raises a 400. Everything else accepts it.
    """
    return not model_name.startswith(("gpt-5", "o1", "o3", "o4"))


def get_model(node: str, **overrides):
    """
    Chat model for a node.

    Deterministic (temperature=0) wherever the model allows it. Sampling variety
    is a liability here: the same question from two customers should get the same
    answer, and the tests need the classifiers to be repeatable.
    """
    name = _model_name(node)
    backend = config.model_backend()

    # A customer is watching a typing indicator, so the timeout is much shorter
    # than the internal agent's 90s — better a fast apology than a minute of
    # silence.
    kwargs: dict = {"timeout": 30, "max_retries": 2}
    if _supports_temperature(name):
        kwargs["temperature"] = 0
    kwargs.update(overrides)

    if backend == "openrouter":
        # ChatOpenRouter, not ChatOpenAI+base_url: the latter targets the
        # official OpenAI spec and drops OpenRouter's routing fields.
        return init_chat_model(
            name,
            model_provider="openrouter",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            **kwargs,
        )

    return init_chat_model(name, model_provider="openai", **kwargs)


def describe() -> str:
    """Human-readable dump of what each node will actually call."""
    lines = [f"backend: {config.model_backend()}"]
    for node in REGISTRY:
        lines.append(f"  {node:10} {_model_name(node):26} — {REGISTRY[node].rationale}")
    return "\n".join(lines)
