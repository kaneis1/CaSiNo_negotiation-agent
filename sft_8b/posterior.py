"""Monte-Carlo posterior over opponent priority orderings.

``get_posterior`` wraps the SFT'd 8B as a probabilistic opponent-model:
sample K completions at temperature T, parse each sample's ``prefs``
field (the [top, mid, low] permutation the model is trained to emit),
count orderings, normalize. Returns a length-6 probability vector over
``ORDERINGS`` — the canonical enumeration of the 6 permutations of
(Food, Water, Firewood).

This vector is what the menu builder in ``sft_8b/menu.py`` consumes
to compute ``E_θ[U_opp(π | θ)]`` over the 64 candidate splits. The
wrapper is deliberately thin and model-agnostic on the *output* side
— any posterior over ``ORDERINGS`` plugs into the menu builder, not
just this SFT-derived one.
"""

from __future__ import annotations

import logging
from itertools import permutations
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from sft_8b.predict import SftModelFn, parse_response
from sft_8b.prompts import ITEMS, build_user_prompt

logger = logging.getLogger("sft_8b.posterior")

# Canonical enumeration: ORDERINGS[i] is the i-th (top, mid, low) tuple.
ORDERINGS: List[Tuple[str, str, str]] = list(permutations(ITEMS))
N_ORDERINGS: int = len(ORDERINGS)  # 6
_ORDERING_INDEX = {o: i for i, o in enumerate(ORDERINGS)}


def get_posterior(
    dialogue_prefix: Sequence[Mapping[str, Any]],
    speaker_priorities: Mapping[str, str],
    *,
    model_fn: SftModelFn,
    speaker_reasons: Optional[Mapping[str, str]] = None,
    me_role: str = "mturk_agent_1",
    K: int = 16,
    temperature: float = 0.7,
) -> np.ndarray:
    """Sample ``K`` completions from the SFT model and return a posterior.

    Parameters
    ----------
    dialogue_prefix
        Turns of the CaSiNo chat log up to (but not including) the point
        we want a posterior for. Same format ``sft_8b.data`` uses when
        it builds training snapshots: each turn is a dict with ``id``
        (role) and ``text`` fields.
    speaker_priorities
        ``{"High": "Food", "Medium": "Water", "Low": "Firewood"}`` —
        the speaker's own CaSiNo priority ranking. Required for the
        prompt; has no effect on which posterior vector indexes mean
        what (those are fixed by ``ORDERINGS``).
    model_fn
        A loaded ``SftModelFn``. Reused across calls so the base weights
        + adapter are loaded exactly once.
    speaker_reasons
        Optional ``value2reason`` dict. Matches what the SFT model saw
        at training time; omitting it renders "(no reasons provided)"
        in the prompt — slight distribution shift so prefer to pass it.
    me_role
        Which side of the chat_logs is "Me:" when rendering the dialogue.
        Must match the role whose ``participant_info`` ``speaker_priorities``
        came from.
    K, temperature
        Sampling budget. The defaults (K=16, T=0.7) are the
        roadmap-committed setting. Increase K if your K-ordering mass
        estimates look jittery turn-to-turn; T=0.7 is tuned against
        the T=0.0 SFT eval — warmer would trade peakedness for spurious
        orderings the base model assigns non-trivial residual mass to.
    """
    prompt = build_user_prompt(
        partial=dialogue_prefix,
        my_priorities=speaker_priorities,
        my_reasons=speaker_reasons or {},
        me_role=me_role,
    )
    raw_samples = model_fn.generate_raw(prompt, K=K, temperature=temperature)

    counts = np.zeros(N_ORDERINGS, dtype=np.float64)
    n_parsed = 0
    n_unknown_ordering = 0
    for raw in raw_samples:
        prefs, _sat, flags = parse_response(raw)
        if flags.get("prefs_malformed") or flags.get("json_malformed"):
            continue
        idx = _ORDERING_INDEX.get(tuple(prefs))
        if idx is None:
            n_unknown_ordering += 1
            continue
        counts[idx] += 1
        n_parsed += 1

    if n_parsed == 0:
        logger.warning(
            "get_posterior: %d/%d samples parsed as a valid ordering; "
            "returning uniform.", n_parsed, K,
        )
        return np.full(N_ORDERINGS, 1.0 / N_ORDERINGS)

    if n_unknown_ordering:
        logger.debug("get_posterior: %d sample(s) parsed JSON but not an "
                     "ordering permutation; skipped.", n_unknown_ordering)

    return counts / n_parsed


def entropy(posterior: np.ndarray, *, base: float = 2.0, eps: float = 1e-12) -> float:
    """Shannon entropy of a posterior vector (in bits by default)."""
    p = np.asarray(posterior, dtype=np.float64)
    p = p[p > eps]
    return float(-(p * (np.log(p) / np.log(base))).sum())


__all__ = ["ORDERINGS", "N_ORDERINGS", "get_posterior", "entropy"]
