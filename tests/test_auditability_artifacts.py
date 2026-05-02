import unittest

import numpy as np

from opponent_model.scripts.make_auditability_artifacts import (
    action_alignment,
    action_or_bid_changed,
    canonical_action_type,
    case_tags_for_row,
    confidence_bins,
    index_records,
    normalize_posterior,
    posterior_confidence,
    posterior_entropy_bits,
    prefix_change_rows,
    record_key,
    select_belief_policy_cases,
)


class AuditabilityArtifactTests(unittest.TestCase):
    def test_posterior_entropy_and_confidence(self):
        posterior = normalize_posterior([2, 2, 0, 0, 0, 0])
        self.assertIsNotNone(posterior)
        self.assertAlmostEqual(posterior_confidence(posterior), 0.5)
        self.assertAlmostEqual(posterior_entropy_bits(posterior), 1.0)

    def test_action_alignment_accept_and_bid(self):
        self.assertEqual(canonical_action_type({"action": "accept", "accept": True}), "accept")
        self.assertTrue(
            action_alignment(
                "accept",
                "accept",
                student_bid=None,
                reference_bid=None,
                bid_close_threshold=0.90,
            )
        )
        self.assertFalse(
            action_alignment(
                "utter",
                "accept",
                student_bid=None,
                reference_bid=None,
                bid_close_threshold=0.90,
            )
        )
        self.assertTrue(
            action_alignment(
                "bid",
                "bid",
                student_bid=np.array([2, 1, 0, 1, 2, 3], dtype=float),
                reference_bid=np.array([2, 1, 0, 1, 2, 3], dtype=float),
                bid_close_threshold=0.90,
            )
        )
        self.assertFalse(
            action_alignment(
                "bid",
                "bid",
                student_bid=None,
                reference_bid=np.array([2, 1, 0, 1, 2, 3], dtype=float),
                bid_close_threshold=0.90,
            )
        )

    def test_case_tags_are_overlapping_diagnostics(self):
        row = {
            "belief_correct": False,
            "menu_alignment": True,
            "human_agreement": True,
            "correct_menu_alignment": False,
        }
        tags = case_tags_for_row(row)
        self.assertIn("belief wrong / policy consistent", tags)
        self.assertIn("belief wrong / lucky action", tags)

        full_failure = {
            "belief_correct": False,
            "menu_alignment": False,
            "human_agreement": False,
            "correct_menu_alignment": False,
        }
        self.assertEqual(case_tags_for_row(full_failure), ["full failure"])

    def test_record_key_index_and_prefix_change_rows(self):
        base = {
            "dialogue_id": "d1",
            "perspective": "mturk_agent_1",
            "turn_index": 4,
            "turn_text": "Accept-Deal",
            "pred": {"action": "utter", "accept": None, "bid": None},
            "true": {"accept": True, "bid": None},
        }
        pref = {
            "dialogue_id": "d1",
            "perspective": "mturk_agent_1",
            "turn_index": 4,
            "turn_text": "Accept-Deal",
            "pred": {"action": "accept", "accept": True, "bid": None},
            "true": {"accept": True, "bid": None},
        }
        self.assertEqual(record_key(base), ("d1", "mturk_agent_1", 4))
        self.assertIn(record_key(base), index_records([base]))
        self.assertTrue(action_or_bid_changed(base, pref))

        rows = prefix_change_rows(
            [base],
            [pref],
            prefix_label="correct_prefix",
            analysis_by_key={
                record_key(base): {
                    "human_agreement": False,
                    "belief_correct": False,
                    "map_ordering": "Food > Water > Firewood",
                    "true_ordering": "Water > Food > Firewood",
                }
            },
            bid_close_threshold=0.90,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["agreement_change"], "improved")

    def test_case_selection_and_confidence_bins(self):
        rows = [
            {
                "dialogue_id": "a",
                "perspective": "p",
                "turn_index": 1,
                "audit_supported": True,
                "belief_correct": False,
                "menu_alignment": True,
                "human_agreement": False,
                "correct_menu_alignment": False,
                "case_tags": ["belief wrong / policy consistent"],
                "confidence": 0.9,
                "entropy_bits": 0.2,
            },
            {
                "dialogue_id": "b",
                "perspective": "p",
                "turn_index": 2,
                "audit_supported": True,
                "belief_correct": True,
                "menu_alignment": False,
                "human_agreement": True,
                "correct_menu_alignment": True,
                "case_tags": ["belief right / policy inconsistent"],
                "confidence": 0.4,
                "entropy_bits": 1.8,
            },
        ]
        selected = select_belief_policy_cases(rows, max_per_tag=1)
        self.assertEqual(len(selected), 2)
        bins = confidence_bins(rows)
        self.assertEqual(len(bins), 6)
        self.assertEqual(sum(row["n"] for row in bins), 2)


if __name__ == "__main__":
    unittest.main()
