import math
import unittest
import json
from pathlib import Path

import numpy as np

from opponent_model.hypotheses import HYPOTHESES
from structured_cot.opponent_inference import (
    brier_from_sample_indices,
    loose_action_consistent,
    parse_opponent_inference_response,
    parse_ranking_response,
    posterior_from_sample_indices,
    strict_action_consistent,
)
from structured_cot.run_structured_opp_inference import (
    iter_regime_a_snapshots,
    iter_regime_b_snapshots,
    load_turn_record_keys,
)


VALID_RESPONSE = """<opponent_inference>
<evidence>
1. "I need water for hiking" [Self-Need]
2. Submit-Deal: opponent keeps 2 water.
</evidence>
<interpretation>
Water is likely high; Firewood has no clear signal; Food is least supported.
</interpretation>
<ranking>{"food":3,"water":1,"firewood":2,"confidence":{"food":"low","water":"high","firewood":"medium"}}</ranking>
<rationale>
The opponent's stated need and offer both point to Water as highest.
</rationale>
</opponent_inference>"""


class StructuredOpponentInferenceParserTests(unittest.TestCase):
    def test_parse_valid_nested_response(self):
        parsed = parse_opponent_inference_response(VALID_RESPONSE)
        self.assertIsNone(parsed["parse_error"])
        self.assertEqual(parsed["ranking"], {"Food": 3, "Water": 1, "Firewood": 2})
        self.assertEqual(parsed["ordering"], ["Water", "Firewood", "Food"])
        self.assertEqual(parsed["confidence"]["Water"], "high")
        self.assertIsInstance(parsed["hypothesis_index"], int)

    def test_missing_nested_tag_fails(self):
        raw = VALID_RESPONSE.replace("<rationale>", "").replace("</rationale>", "")
        parsed = parse_opponent_inference_response(raw)
        self.assertIsNotNone(parsed["parse_error"])
        self.assertIn("rationale", parsed["missing_tags"])

    def test_malformed_ranking_json_fails(self):
        raw = VALID_RESPONSE.replace(
            '{"food":3,"water":1,"firewood":2,"confidence":{"food":"low","water":"high","firewood":"medium"}}',
            '{"food":3,"water":1,',
        )
        parsed = parse_opponent_inference_response(raw)
        self.assertIsNotNone(parsed["parse_error"])
        self.assertIn("could not parse ranking JSON object", parsed["ranking_errors"])

    def test_duplicate_ranks_fail(self):
        raw = VALID_RESPONSE.replace('"firewood":2', '"firewood":1')
        parsed = parse_opponent_inference_response(raw)
        self.assertIsNotNone(parsed["parse_error"])
        self.assertTrue(
            any("ranking values must be exactly" in err for err in parsed["ranking_errors"])
        )

    def test_bad_confidence_fails(self):
        raw = VALID_RESPONSE.replace('"water":"high"', '"water":"certain"')
        parsed = parse_opponent_inference_response(raw)
        self.assertIsNotNone(parsed["parse_error"])
        self.assertTrue(
            any("confidence for Water" in err for err in parsed["ranking_errors"])
        )

    def test_parse_standalone_ranking_response(self):
        raw = """<ranking>{"food":2,"water":1,"firewood":3,"confidence":{"food":"medium","water":"high","firewood":"low"}}</ranking>"""
        parsed = parse_ranking_response(raw)
        self.assertIsNone(parsed["parse_error"])
        self.assertEqual(parsed["ordering"], ["Water", "Food", "Firewood"])


class StructuredOpponentInferenceMetricTests(unittest.TestCase):
    def test_invalid_self_consistency_sample_contributes_uniform_mass(self):
        posterior = posterior_from_sample_indices([0, 0, None, None, None])
        self.assertAlmostEqual(float(posterior.sum()), 1.0)
        self.assertGreater(posterior[0], 2.0 / 5.0)
        self.assertTrue(np.all(posterior > 0.0))

    def test_brier_from_sample_indices_is_finite(self):
        score = brier_from_sample_indices([0, 0, None, 1, 2], true_index=0)
        self.assertTrue(math.isfinite(score))

    def test_action_consistency_helpers(self):
        my_priorities = {"High": "Food", "Medium": "Water", "Low": "Firewood"}
        predicted = ["Water", "Firewood", "Food"]
        self.assertTrue(
            strict_action_consistent(
                offer_self_counts={"Food": 3, "Water": 0, "Firewood": 0},
                my_priorities=my_priorities,
                predicted_ordering=predicted,
                lambda_=1.0,
            )
        )
        self.assertTrue(
            loose_action_consistent(
                offer_self_counts={"Food": 3, "Water": 0, "Firewood": 1},
                predicted_ordering=predicted,
            )
        )


class StructuredOpponentInferenceSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with Path("data/casino_test.json").open() as f:
            cls.dialogues = json.load(f)

    def test_regime_a_prefix_support_matches_split(self):
        snaps = list(
            iter_regime_a_snapshots(
                self.dialogues,
                perspective="mturk_agent_1",
                max_dialogues=150,
            )
        )
        self.assertEqual(len(snaps), 746)
        self.assertEqual(
            {k: sum(1 for s in snaps if s["k"] == k) for k in range(1, 6)},
            {1: 150, 2: 150, 3: 150, 4: 149, 5: 147},
        )

    def test_regime_b_matches_teacher_turn_records(self):
        path = Path(
            "opponent_model/results/turn_eval_bayesian_lambda1.0_m5_f0.50_full150/"
            "turn_records.jsonl"
        )
        if not path.exists():
            self.skipTest(f"missing matched turn records: {path}")
        records = load_turn_record_keys(path, perspective="mturk_agent_1")
        snaps = list(iter_regime_b_snapshots(self.dialogues, matched_turn_records=records))
        self.assertEqual(len(records), 1054)
        self.assertEqual(len(snaps), 1054)


if __name__ == "__main__":
    unittest.main()
