"""Structured Chain-of-Thought negotiator for CaSiNo.

The agent produces five XML-tagged reasoning blocks per turn —
``<observation>``, ``<opponent_inference>``, ``<plan>``, ``<utterance>``,
``<decision>`` — so that each decision is accompanied by an auditable
trace. This baseline is the starting point for the distilled 8B
student; its logs feed Protocol 3 evaluation.
"""

from structured_cot.agent import StructuredCoTAgent
from structured_cot.llm_client import DummyStructuredLLM, StructuredLLMClient
from structured_cot.parser import (
    REMINDER,
    parse_response,
    safe_default,
    validate_decision,
)
from structured_cot.prompts import build_prompt
from structured_cot.retrieval_opponent import (
    RetrievalOpponent,
    build_retrieval_pool,
    load_training_corpus,
    pareto_max_self,
    points_for,
)

__all__ = [
    "StructuredCoTAgent",
    "StructuredLLMClient",
    "DummyStructuredLLM",
    "REMINDER",
    "build_prompt",
    "parse_response",
    "safe_default",
    "validate_decision",
    "RetrievalOpponent",
    "build_retrieval_pool",
    "load_training_corpus",
    "pareto_max_self",
    "points_for",
]
