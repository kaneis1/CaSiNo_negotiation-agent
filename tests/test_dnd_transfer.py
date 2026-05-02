import unittest

import numpy as np

from sft_8b.dnd_data import (
    DND_ITEMS,
    canonical_item,
    compute_stats,
    context_total_value,
    map_text_names,
    parse_dnd_line,
    values_to_ordering,
)
from sft_8b.dnd_menu import (
    build_dnd_menu,
    build_value_map_543,
    enumerate_allocations,
    normalize_value_map_for_counts,
    utility,
)
from sft_8b.dnd_metrics import brier_reference, normalized_brier_sum, summarize_snapshot_metrics
from sft_8b.dnd_posterior import ORDERINGS, ORDERING_INDEX, parse_posterior_response, parse_prefs_response
from sft_8b.dnd_rules import posterior_from_evidence, score_utterance


SAMPLE = (
    "<input> 2 2 3 2 1 0 </input> "
    "<dialogue> THEM: i need that ball so bad ! what do you want ? <eos> "
    "YOU: i mean i'll take the rest <eos> "
    "THEM: could i also have one hat maybe ? pretty please ? <eos> "
    "YOU: you drive a hard bargain here , ball and a book ? <eos> "
    "THEM: if that's the offer , then you just take the book because they have no value for me . <eos> "
    "YOU: <selection> </dialogue> "
    "<output> item0=2 item1=3 item2=0 item0=0 item1=0 item2=1 </output> "
    "<partner_input> 2 0 3 1 1 7 </partner_input>"
)

STRICT_SAMPLE = (
    "<input> 2 4 2 1 1 0 </input> "
    "<dialogue> YOU: i want one book <eos> THEM: i get all hats <eos> YOU: <selection> </dialogue> "
    "<output> item0=1 item1=0 item2=1 item0=1 item1=2 item2=0 </output> "
    "<partner_input> 2 0 2 4 1 2 </partner_input>"
)


class DNDTransferTests(unittest.TestCase):
    def test_parse_dnd_line_and_orientation(self):
        rec = parse_dnd_line(SAMPLE, split="test", line_index=0)
        self.assertEqual(rec.counts, (2, 3, 1))
        self.assertEqual(rec.self_values, (2, 2, 0))
        self.assertEqual(rec.partner_values, (0, 1, 7))
        self.assertIsNone(rec.self_ordering)
        self.assertEqual(rec.partner_ordering, ("balls", "hats", "books"))
        self.assertEqual(rec.output_self, (2, 3, 0))
        self.assertEqual(rec.output_partner, (0, 0, 1))
        self.assertEqual(rec.selection_speaker, "YOU")
        self.assertEqual(context_total_value(rec.counts, rec.self_values), 10.0)
        self.assertEqual(context_total_value(rec.counts, rec.partner_values), 10.0)

    def test_parse_rejects_non_10_point_context(self):
        bad = SAMPLE.replace("<input> 2 2 3 2 1 0 </input>", "<input> 2 3 3 2 1 0 </input>")
        with self.assertRaisesRegex(ValueError, "total 10 points"):
            parse_dnd_line(bad, split="test", line_index=0)

    def test_tie_detection_and_stats(self):
        tied = parse_dnd_line(SAMPLE, split="test", line_index=0)
        strict = parse_dnd_line(STRICT_SAMPLE, split="test", line_index=1)
        self.assertEqual(values_to_ordering((5, 3, 2)), ("books", "hats", "balls"))
        self.assertIsNone(values_to_ordering((5, 5, 0)))
        stats = compute_stats([tied, strict])
        self.assertEqual(stats["strict_partner"], 2)
        self.assertEqual(stats["strict_both"], 1)
        self.assertEqual(stats["self_tie"], 1)

    def test_item_aliases_and_renaming(self):
        self.assertEqual(canonical_item("Food"), "books")
        self.assertEqual(canonical_item("fire wood"), "balls")
        self.assertEqual(
            map_text_names("I need books and one hat, you take the ball.", name_mode="renamed"),
            "I need food and one water, you take the firewood.",
        )

    def test_prefs_only_parser_accepts_legacy_satisfaction(self):
        ordering, errors = parse_prefs_response('{"prefs":["Food","Water","Firewood"],"satisfaction":"Undecided"}')
        self.assertFalse(errors)
        self.assertEqual(ordering, ("books", "hats", "balls"))

    def test_posterior_parser_accepts_native_block(self):
        raw = "<posterior>\n" + "\n".join(
            f"p({' > '.join(o)})={1.0 if i == 0 else 0.0}"
            for i, o in enumerate(ORDERINGS)
        ) + "\n</posterior>"
        posterior, errors = parse_posterior_response(raw)
        self.assertFalse(errors)
        self.assertAlmostEqual(float(posterior.sum()), 1.0)
        self.assertEqual(int(np.argmax(posterior)), 0)

    def test_rule_offer_evidence(self):
        counts = {"books": 2, "hats": 3, "balls": 1}
        ev = score_utterance("I get all books and one hat. You can have the ball.", counts=counts)
        posterior = posterior_from_evidence(ev, ORDERINGS)
        books_high = sum(posterior[i] for i, o in enumerate(ORDERINGS) if o[0] == "books")
        balls_low = sum(posterior[i] for i, o in enumerate(ORDERINGS) if o[2] == "balls")
        self.assertGreater(books_high, 1.0 / 3.0)
        self.assertGreater(balls_low, 1.0 / 3.0)

    def test_menu_enumeration_and_scoring(self):
        rec = parse_dnd_line(STRICT_SAMPLE, split="test", line_index=1)
        allocs = list(enumerate_allocations(rec.counts))
        self.assertEqual(len(allocs), (2 + 1) * (2 + 1) * (1 + 1))
        p = np.full(len(ORDERINGS), 1.0 / len(ORDERINGS))
        menu = build_dnd_menu(
            posterior=p,
            orderings=ORDERINGS,
            counts=rec.counts,
            self_values=rec.self_values,
            opp_value_map=build_value_map_543(ORDERINGS),
            lambda_=1.0,
        )
        self.assertTrue(menu)
        self.assertEqual(set(menu[0].self_counts), set(DND_ITEMS))

    def test_opponent_value_map_is_normalized_to_dnd_budget(self):
        rec = parse_dnd_line(STRICT_SAMPLE, split="test", line_index=1)
        normalized = normalize_value_map_for_counts(build_value_map_543(ORDERINGS), rec.counts, ORDERINGS)
        for ordering in ORDERINGS:
            self.assertAlmostEqual(utility(rec.counts, normalized[ordering]), 10.0)

    def test_brier_reference_and_summary(self):
        true = ORDERING_INDEX[("books", "hats", "balls")]
        uniform = [1.0 / 6.0] * 6
        self.assertAlmostEqual(brier_reference(6), 1.0 / 6.0)
        self.assertAlmostEqual(normalized_brier_sum(uniform, true), 1.0 / 6.0)
        summary = summarize_snapshot_metrics([
            {"k": 1, "ema": 1.0, "top1": 1.0, "ndcg": 1.0, "brier": 0.0},
            {"k": 2, "ema": 0.0, "top1": 1.0, "ndcg": 0.5, "brier": 0.2},
            {"k": 3, "ema": 1.0, "top1": 1.0, "ndcg": 1.0, "brier": 0.0},
        ])
        self.assertEqual(summary["support_by_k"]["1"], 1)
        self.assertAlmostEqual(summary["ema_at2"], 0.0)
        self.assertTrue(np.isfinite(summary["kpenalty_1_3"]["ema"]))


if __name__ == "__main__":
    unittest.main()
