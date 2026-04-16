#!/usr/bin/env python3
"""Bidding strategy module for CaSiNo negotiation.

Implements a Boulware concession curve that starts with ambitious asks and
concedes slowly toward a reservation value.  Integrates with the opponent
model to exploit integrative bargaining potential when priorities differ.
"""

from __future__ import annotations

from typing import Any, Dict

from prompt_engineer.preprocessing.scoring import PRIORITY_POINTS, WALK_AWAY_POINTS

ITEMS = ("Food", "Water", "Firewood")
PACKAGES_PER_ITEM = 3
TOTAL_PACKAGES = PACKAGES_PER_ITEM * len(ITEMS)  # 9


class BiddingStrategy:
    """Boulware-curve bidder that concedes slowly from aspiration to reservation.

    Args:
        my_priorities: {"High": "Food", "Medium": "Water", "Low": "Firewood"}.
        beta: Boulware exponent (higher = more stubborn). Default 3.0.
        max_turns: Expected max conversation turns. Default 10.
    """

    def __init__(
        self,
        my_priorities: Dict[str, str],
        beta: float = 3.0,
        max_turns: int = 10,
    ) -> None:
        self.my_priorities = my_priorities
        self.priority_to_item = my_priorities
        self.item_to_priority = {v: k for k, v in my_priorities.items()}
        self.beta = beta
        self.max_turns = max_turns

        high = self.priority_to_item["High"]
        med = self.priority_to_item["Medium"]
        low = self.priority_to_item["Low"]

        self.aspiration = {high: 3, med: 2, low: 0}
        self.reservation = {high: 2, med: 1, low: 0}

    # ── Boulware concession curve ──────────────────────────────────────

    def get_target(self, turn: int) -> Dict[str, int]:
        """Target allocation at *turn* along the Boulware curve.

        Early turns stay close to aspiration; concession accelerates near
        max_turns.  Returns counts that respect 0 <= count <= 3 per item
        and total <= 9.
        """
        t_ratio = min(turn / self.max_turns, 1.0)
        concession = 1 - (1 - t_ratio) ** (1 / self.beta)

        target: Dict[str, int] = {}
        for item in ITEMS:
            asp = self.aspiration.get(item, 0)
            res = self.reservation.get(item, 0)
            target[item] = max(0, min(3, round(asp - concession * (asp - res))))

        total = sum(target.values())
        if total > TOTAL_PACKAGES:
            for priority in ("Low", "Medium", "High"):
                item = self.priority_to_item[priority]
                while total > TOTAL_PACKAGES and target[item] > 0:
                    target[item] -= 1
                    total -= 1

        return target

    # ── Offer generation ───────────────────────────────────────────────

    def generate_offer(
        self,
        turn: int,
        opponent_model: Any = None,
    ) -> Dict[str, Dict[str, int]]:
        """Generate a concrete deal proposal.

        If an opponent model with medium/high confidence is available and
        both sides have different High-priority items, exploit the
        integrative potential by trading low-value items.

        Returns:
            {"me": {item: count, ...}, "them": {item: 3 - count, ...}}
        """
        my_target = self.get_target(turn)

        if opponent_model and opponent_model.confidence != "low":
            opp_priorities = opponent_model.get_predicted_priorities()
            my_high = self.priority_to_item["High"]
            opp_high = opp_priorities["High"]

            if my_high != opp_high:
                my_target[opp_high] = max(0, my_target[opp_high] - 1)
                my_target[my_high] = min(3, my_target[my_high] + 1)

        deal = {
            "me": dict(my_target),
            "them": {item: PACKAGES_PER_ITEM - count
                     for item, count in my_target.items()},
        }
        return deal

    # ── Offer evaluation ───────────────────────────────────────────────

    def score_allocation(self, allocation: Dict[str, int]) -> int:
        """Points scored from an allocation using own priority weights."""
        return sum(
            int(count) * PRIORITY_POINTS[self.item_to_priority[item]]
            for item, count in allocation.items()
        )

    def evaluate_offer(
        self,
        their_offer_for_me: Dict[str, int],
        turn: int,
    ) -> bool:
        """Decide whether to accept an incoming offer.

        Accepts if the offer meets the current Boulware target, or if
        we're near the deadline and the score is still reasonable
        (above walk-away).
        """
        my_score = self.score_allocation(their_offer_for_me)
        target_score = self.score_allocation(self.get_target(turn))

        if my_score >= target_score:
            return True
        if turn >= self.max_turns - 2 and my_score >= WALK_AWAY_POINTS * 2 + 2:
            return True
        return False

    # ── Utilities ──────────────────────────────────────────────────────

    def format_offer_text(self, deal: Dict[str, Dict[str, int]]) -> str:
        """Render a deal as natural-language text for prompt injection."""
        me = deal["me"]
        them = deal["them"]
        my_parts = [f"{c} {item}" for item, c in me.items() if c > 0]
        their_parts = [f"{c} {item}" for item, c in them.items() if c > 0]
        return (
            f"I'd like {', '.join(my_parts)}. "
            f"You can have {', '.join(their_parts)}."
        )

    def format_for_submit_deal(
        self,
        deal: Dict[str, Dict[str, int]],
    ) -> Dict[str, Dict[str, str]]:
        """Convert deal to CaSiNo Submit-Deal task_data format."""
        return {
            "issue2youget": {item: str(count) for item, count in deal["me"].items()},
            "issue2theyget": {item: str(count) for item, count in deal["them"].items()},
        }

    def summary(self, turn: int) -> Dict[str, Any]:
        """Snapshot of current bidding state for debugging."""
        target = self.get_target(turn)
        return {
            "turn": turn,
            "target": target,
            "target_score": self.score_allocation(target),
            "aspiration_score": self.score_allocation(self.aspiration),
            "reservation_score": self.score_allocation(self.reservation),
            "walk_away_score": WALK_AWAY_POINTS,
        }


# ── CLI demo ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    priorities = {"High": "Food", "Medium": "Water", "Low": "Firewood"}
    bidder = BiddingStrategy(priorities, beta=3.0, max_turns=10)

    print(f"Priorities: {priorities}")
    print(f"Aspiration: {bidder.aspiration}  "
          f"(score={bidder.score_allocation(bidder.aspiration)})")
    print(f"Reservation: {bidder.reservation}  "
          f"(score={bidder.score_allocation(bidder.reservation)})")
    print(f"Walk-away: {WALK_AWAY_POINTS}")
    print()

    header = f"{'Turn':>4}  {'Food':>4}  {'Water':>5}  {'Firewood':>8}  {'Score':>5}  {'Offer text'}"
    print(header)
    print("─" * len(header))

    for t in range(0, 11):
        deal = bidder.generate_offer(t)
        score = bidder.score_allocation(deal["me"])
        text = bidder.format_offer_text(deal)
        me = deal["me"]
        print(f"{t:>4}  {me['Food']:>4}  {me['Water']:>5}  {me['Firewood']:>8}  "
              f"{score:>5}  {text}")
