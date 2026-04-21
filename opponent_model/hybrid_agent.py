"""Hybrid LLM + Bayesian opponent-modeling agent.

The overall loop, per the design spec:

1. ``observe(opponent_utterance)``: ask the LLM for a likelihood vector
   (relative evidence, no prior), then do a pure-Python Bayes update in
   log space.
2. ``speak()``: render a natural-language posterior summary, ask the LLM
   for the next utterance + (optional) offer.

LLM I/O is the only place we touch the model. Everything else is closed-form.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from opponent_model.big_five_prior import big_five_prior
from opponent_model.hypotheses import HYPOTHESES, ITEMS, hypothesis_label

logger = logging.getLogger(__name__)


class MalformedLikelihoodError(ValueError):
    """Raised when the LLM's likelihood JSON is missing hypothesis scores.

    Carries the names of the missing hypotheses (e.g. ``["H3", "H5"]``) and
    the raw LLM response so the caller can either fix the prompt or
    explicitly opt into the silent-fallback behavior.
    """

    def __init__(self, missing: List[str], raw_response: str) -> None:
        self.missing = list(missing)
        self.raw_response = raw_response
        super().__init__(
            f"LLM evidence JSON is missing scores for hypotheses {self.missing}. "
            f"Defaulting these to 50 would silently treat them as 'no evidence', "
            f"but the prompt probably needs fixing first. Raw response:\n{raw_response!r}"
        )
from opponent_model.prompts import (
    build_generation_prompt,
    build_likelihood_prompt,
)

NUM_HYPOTHESES = len(HYPOTHESES)


# ── Small numerical helpers (stay numpy-only, no scipy dep) ────────────────


def _logsumexp(x: np.ndarray) -> float:
    """Numerically stable log-sum-exp over a 1-D array."""
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))


def _safe_json_loads(text: str) -> Dict[str, Any]:
    """Best-effort JSON parsing for LLM output.

    LLMs often wrap JSON in prose or markdown fences. We extract the first
    balanced ``{...}`` block and parse that.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in LLM output:\n{text!r}")
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = text[start : i + 1]
                return json.loads(blob)
    raise ValueError(f"Unbalanced JSON braces in LLM output:\n{text!r}")


# ── HybridAgent ────────────────────────────────────────────────────────────


class HybridAgent:
    """LLM-likelihood + Bayesian-update opponent model + LLM-conditioned speaker.

    Args:
        my_priorities: ``{"High": "Food", "Medium": "Water", "Low": "Firewood"}``.
        llm_client: Object with ``.generate(prompt: str) -> str``.
        big_five: Optional Big Five dict to seed the prior. ``None`` → uniform.
        big_five_bumps: Optional regression-coefficient table for the
            (currently stub) Big Five → Dirichlet mapping. See
            ``opponent_model.big_five_prior.big_five_prior``.
        prior_concentration: Pseudo-count strength of the Big Five prior.
        certainty_thresholds: ``(low, mid)`` cutoffs for posterior summary
            wording. Defaults match the spec: 0.15 / 0.50.
        strict_likelihood: If True (default), raise ``MalformedLikelihoodError``
            when the LLM omits any of the 6 hypothesis scores. Use this while
            iterating on the prompt — silent fallback to neutral (50) hides
            real prompt bugs (a missing key under "50 = neutral" is *not* the
            same as "50 was meant"; the LLM might have *intended* extreme
            evidence and just dropped the field). Flip to False once the
            prompt is stable; the agent will then warn-and-fill-with-50.
        likelihood_temperature: Divisor in the score → log-likelihood map
            ``log_lik = (score - 50) / T``. Smaller T ⇒ each LLM-evidence
            point moves the posterior more aggressively. Default 25.0 maps
            the prompt's recommended 15–85 band to roughly ±1.4 nats per
            turn (≈4× likelihood ratio). Sweep this during calibration.
        likelihood_clip: ``(low, high)`` clamp applied to the centered
            log-likelihood *before* the max-subtract normalization. Caps
            how confident a single utterance can be after temperature
            scaling. Default ``(-3.0, 3.0)`` ≈ a 20× ratio cap. Pass
            ``(None, None)`` to disable clipping during a sweep.
    """

    def __init__(
        self,
        my_priorities: Mapping[str, str],
        llm_client: Any,
        big_five: Optional[Mapping[str, float]] = None,
        big_five_bumps: Optional[Mapping[str, Mapping[str, float]]] = None,
        prior_concentration: float = 6.0,
        certainty_thresholds: Tuple[float, float] = (0.15, 0.50),
        strict_likelihood: bool = True,
        likelihood_temperature: float = 25.0,
        likelihood_clip: Tuple[Optional[float], Optional[float]] = (-3.0, 3.0),
    ) -> None:
        if likelihood_temperature <= 0.0:
            raise ValueError(
                f"likelihood_temperature must be > 0 (got {likelihood_temperature})."
            )
        clip_lo, clip_hi = likelihood_clip
        if (clip_lo is not None and clip_hi is not None) and clip_lo >= clip_hi:
            raise ValueError(
                f"likelihood_clip lower bound must be < upper bound "
                f"(got {likelihood_clip})."
            )

        self.my_priorities: Dict[str, str] = dict(my_priorities)
        self.llm_client = llm_client
        self.hypotheses = HYPOTHESES
        self.log_posterior: np.ndarray = self._init_prior(
            big_five, big_five_bumps, prior_concentration
        )
        self.history: List[Dict[str, str]] = []  # {"role": "opp"|"me", "text": ...}
        self._certainty_low, self._certainty_mid = certainty_thresholds
        self._update_log: List[Dict[str, Any]] = []
        self.strict_likelihood = strict_likelihood
        self.likelihood_temperature = float(likelihood_temperature)
        self.likelihood_clip: Tuple[Optional[float], Optional[float]] = (
            None if clip_lo is None else float(clip_lo),
            None if clip_hi is None else float(clip_hi),
        )

    # ── Loop step 1 ────────────────────────────────────────────────────

    def observe(self, opponent_utterance: str) -> Dict[str, Any]:
        """Ingest an opponent utterance: score with LLM, Bayes-update posterior.

        Returns a small debug dict (likelihood, rationale, new posterior).
        """
        self.history.append({"role": "opp", "text": opponent_utterance})

        log_lik, rationale, raw = self._llm_likelihood(opponent_utterance)
        self.log_posterior = self._bayes_update(self.log_posterior, log_lik)

        record = {
            "utterance": opponent_utterance,
            "log_likelihood": log_lik.tolist(),
            "rationale": rationale,
            "raw": raw,
            "log_posterior": self.log_posterior.tolist(),
        }
        self._update_log.append(record)
        return record

    # ── Loop step 2 ────────────────────────────────────────────────────

    def speak(self) -> Tuple[str, Optional[Dict[str, int]]]:
        """Generate the next utterance + (optional) offer.

        Returns:
            ``(utterance, offer)`` where ``offer`` is a ``{item: count}``
            dict or ``None``.
        """
        summary = self._posterior_summary()
        utterance, offer = self._llm_generate(summary)
        self.history.append({"role": "me", "text": utterance})
        return utterance, offer

    # ── Piece 1: LLM likelihood ────────────────────────────────────────

    def _llm_likelihood(
        self,
        opponent_utterance: str,
    ) -> Tuple[np.ndarray, str, str]:
        """Ask the LLM for relative evidence under each of the 6 hypotheses.

        Returns ``(log_lik, rationale, raw_response)``.

        The prompt asks the LLM for evidence on a 0–100 scale where 50 means
        "the utterance carries no information for or against this hypothesis."
        We center on 50, divide by ``self.likelihood_temperature`` to convert
        into a relative log-likelihood, then clamp to ``self.likelihood_clip``.
        The Bayes update is invariant to additive constants on the
        likelihood, so we additionally subtract the max for numerical
        stability.

        Score → log-likelihood mapping (with the defaults T=25, clip=(-3, 3)):
            85 (strong evidence for)   → +1.40 nats  (~4.0× more likely)
            65 (mild   evidence for)   → +0.60 nats  (~1.8× more likely)
            50 (no signal)             →  0.00 nats
            35 (mild   evidence against)→ -0.60 nats
            15 (strong evidence against)→ -1.40 nats
        Both ``likelihood_temperature`` and ``likelihood_clip`` are
        constructor args so they can be swept during calibration.
        """
        prompt = build_likelihood_prompt(opponent_utterance, history=self.history[:-1])
        raw = self.llm_client.generate(prompt)
        parsed = _safe_json_loads(raw)

        evidence = parsed.get("evidence_scores", {})
        if not isinstance(evidence, Mapping):
            raise MalformedLikelihoodError(
                missing=[hypothesis_label(i) for i in range(NUM_HYPOTHESES)],
                raw_response=raw,
            )

        missing = [
            hypothesis_label(i)
            for i in range(NUM_HYPOTHESES)
            if hypothesis_label(i) not in evidence or evidence[hypothesis_label(i)] is None
        ]
        if missing:
            if self.strict_likelihood:
                raise MalformedLikelihoodError(missing=missing, raw_response=raw)
            logger.warning(
                "LLM evidence JSON missing %s; defaulting to neutral (50). "
                "Raw response: %r", missing, raw,
            )

        scores_list: List[float] = []
        for i in range(NUM_HYPOTHESES):
            value = evidence.get(hypothesis_label(i))
            try:
                scores_list.append(float(value) if value is not None else 50.0)
            except (TypeError, ValueError):
                if self.strict_likelihood:
                    raise MalformedLikelihoodError(
                        missing=[hypothesis_label(i)], raw_response=raw,
                    )
                logger.warning(
                    "LLM gave non-numeric score %r for %s; defaulting to 50.",
                    value, hypothesis_label(i),
                )
                scores_list.append(50.0)
        scores = np.clip(np.array(scores_list, dtype=float), 0.0, 100.0)

        log_lik = (scores - 50.0) / self.likelihood_temperature
        clip_lo, clip_hi = self.likelihood_clip
        if clip_lo is not None or clip_hi is not None:
            log_lik = np.clip(log_lik, clip_lo, clip_hi)
        log_lik = log_lik - log_lik.max()

        rationale = str(parsed.get("short_rationale", "")).strip()
        return log_lik, rationale, raw

    # ── Piece 2: Bayes update (log space) ──────────────────────────────

    def _init_prior(
        self,
        big_five: Optional[Mapping[str, float]],
        big_five_bumps: Optional[Mapping[str, Mapping[str, float]]],
        concentration: float,
    ) -> np.ndarray:
        if big_five is None:
            return np.log(np.ones(NUM_HYPOTHESES) / NUM_HYPOTHESES)
        return big_five_prior(
            big_five, concentration=concentration, bumps=big_five_bumps,
        )

    @staticmethod
    def _bayes_update(log_prior: np.ndarray, log_lik: np.ndarray) -> np.ndarray:
        log_posterior = log_prior + log_lik
        log_posterior = log_posterior - _logsumexp(log_posterior)
        return log_posterior

    # ── Piece 3: Posterior summary → generation ────────────────────────

    def posterior(self) -> np.ndarray:
        """Return the current posterior in probability space, shape (6,)."""
        return np.exp(self.log_posterior)

    def top_marginals(self) -> Dict[str, float]:
        """Marginal probability that each item is the opponent's TOP priority."""
        posterior = self.posterior()
        return {
            item: float(
                sum(posterior[i] for i, h in enumerate(self.hypotheses) if h[0] == item)
            )
            for item in ITEMS
        }

    def certainty(self) -> float:
        """Normalized certainty in ``[0, 1]`` (1 = point mass, 0 = uniform)."""
        posterior = self.posterior()
        entropy = -float(np.sum(posterior * np.log(posterior + 1e-12)))
        max_entropy = float(np.log(NUM_HYPOTHESES))
        return max(0.0, min(1.0, 1.0 - entropy / max_entropy))

    def _posterior_summary(self) -> str:
        top_marginals = self.top_marginals()
        certainty = self.certainty()
        sorted_items = sorted(top_marginals.items(), key=lambda x: -x[1])
        low, mid = self._certainty_low, self._certainty_mid

        if certainty < low:
            return "You have very little information about the opponent's priorities yet."
        if certainty < mid:
            lead_item, lead_p = sorted_items[0]
            return (
                f"The opponent likely values {lead_item} most "
                f"(about {lead_p:.0%} probability), but you're not confident yet."
            )
        lead_item, lead_p = sorted_items[0]
        second_item, _ = sorted_items[1]
        third_item, _ = sorted_items[2]
        return (
            f"The opponent most likely ranks priorities as "
            f"{lead_item} > {second_item} > {third_item} "
            f"(top item confidence: {lead_p:.0%})."
        )

    def _llm_generate(
        self,
        summary: str,
    ) -> Tuple[str, Optional[Dict[str, int]]]:
        prompt = build_generation_prompt(
            my_priorities=self.my_priorities,
            posterior_summary=summary,
            history=self.history,
        )
        raw = self.llm_client.generate(prompt)
        parsed = _safe_json_loads(raw)

        utterance = str(parsed.get("utterance", "")).strip()
        offer_raw = parsed.get("offer", None)
        offer = self._normalize_offer(offer_raw)
        return utterance, offer

    @staticmethod
    def _normalize_offer(offer_raw: Any) -> Optional[Dict[str, int]]:
        """Coerce LLM offer JSON into ``{item: int}`` or ``None``."""
        if offer_raw is None or offer_raw == "null":
            return None
        if not isinstance(offer_raw, Mapping):
            return None
        offer: Dict[str, int] = {}
        for item in ITEMS:
            value = offer_raw.get(item)
            if value is None:
                return None
            try:
                count = int(value)
            except (TypeError, ValueError):
                return None
            offer[item] = max(0, min(3, count))
        return offer

    # ── Misc inspection ────────────────────────────────────────────────

    def predicted_priorities(self) -> Dict[str, str]:
        """Argmax hypothesis rendered as ``{"High": ..., "Medium": ..., "Low": ...}``."""
        idx = int(np.argmax(self.log_posterior))
        h = self.hypotheses[idx]
        return {"High": h[0], "Medium": h[1], "Low": h[2]}

    def state(self) -> Dict[str, Any]:
        """JSON-friendly snapshot for logging/debugging."""
        return {
            "my_priorities": dict(self.my_priorities),
            "posterior": self.posterior().tolist(),
            "top_marginals": self.top_marginals(),
            "certainty": self.certainty(),
            "predicted_priorities": self.predicted_priorities(),
            "history_length": len(self.history),
        }

    def reset(
        self,
        big_five: Optional[Mapping[str, float]] = None,
        big_five_bumps: Optional[Mapping[str, Mapping[str, float]]] = None,
        prior_concentration: float = 6.0,
    ) -> None:
        """Clear history + reseed prior."""
        self.history.clear()
        self._update_log.clear()
        self.log_posterior = self._init_prior(
            big_five, big_five_bumps, prior_concentration,
        )


__all__ = ["HybridAgent", "MalformedLikelihoodError"]
