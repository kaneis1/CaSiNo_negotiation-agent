import json
import unittest
from pathlib import Path

from opponent_model.bid_extractor import extract_bid_from_utterance
from opponent_model.hypotheses import ITEMS


def _opp_counts(self_counts):
    return {item: 3 - int(self_counts[item]) for item in ITEMS}


MANUAL_POSITIVES = [
    (
        "I propose I take 2 food, 1 water, 1 firewood; you take the rest.",
        {"Food": 2, "Water": 1, "Firewood": 1},
    ),
    (
        "Just to confirm, you'll give me all 3 packages of water, and I'll take 1 package of food, and you'll take the rest, including all the firewood.",
        {"Food": 1, "Water": 3, "Firewood": 0},
    ),
    (
        "I could take 2 Firewood and 1 Food, leaving you with 1 Firewood, 2 Food, and the Water packages.",
        {"Food": 1, "Water": 0, "Firewood": 2},
    ),
    (
        "I propose food=2, water=1, firewood=0, and you get the rest.",
        {"Food": 2, "Water": 1, "Firewood": 0},
    ),
    (
        "You can have 1 food, 2 water, and 1 firewood, and I'll take the rest.",
        {"Food": 2, "Water": 1, "Firewood": 2},
    ),
    (
        "For me to get 2 food, 1 water, and 0 firewood, you would get the rest.",
        {"Food": 2, "Water": 1, "Firewood": 0},
    ),
]

MANUAL_NEGATIVES = [
    "I get 1.5 food and you get the rest.",
    "I get 4 water and you get the rest.",
    "I get 1 or 2 food and you get the rest.",
    "I get 2 food, 1 water, and the rest for you, or we can discuss a different split.",
    "I get 2 food and you get 2 water.",
    "I get 2 food, and later I get 1 food, 1 water, and 0 firewood, and you get the rest.",
]

PROTOCOL1_POSITIVES = [
    (5, 0, "fair split would be 2 Food and 1 Water for me", {"Food": 2, "Water": 1, "Firewood": 0}),
    (5, 4, "I get 2 food, 2 water, and 0 firewood", {"Food": 2, "Water": 2, "Firewood": 0}),
    (16, 0, "I get 2 Food and 1 Firewood, and you get the rest", {"Food": 2, "Water": 0, "Firewood": 1}),
    (16, 4, "I get 2 food, 1 water, and 1 firewood", {"Food": 2, "Water": 1, "Firewood": 1}),
    (18, 10, "I get 2 packages of food, 1 package of water, and 1 package of firewood", {"Food": 2, "Water": 1, "Firewood": 1}),
    (33, 0, "I get 2 Food and 1 Water, and you get the rest", {"Food": 2, "Water": 1, "Firewood": 0}),
    (33, 4, "I get 2 Food and 1 Water, and you get 1 Food, 2 Water, and 3 Firewood", {"Food": 2, "Water": 1, "Firewood": 0}),
    (34, 5, "I could take 2 Firewood and 1 Food, leaving you with 1 Firewood, 2 Food, and the Water packages", {"Food": 1, "Water": 0, "Firewood": 2}),
    (34, 11, "I get 2 firewoods and 1 food, and you get 1 firewood, 2 food, and 3 water", {"Food": 1, "Water": 0, "Firewood": 2}),
    (38, 3, "I get 2 firewood and 1 food, and you get the rest", {"Food": 1, "Water": 0, "Firewood": 2}),
]

PROTOCOL1_NEGATIVES = [
    (1, 5, "2 packages of Food and 1 package of Firewood"),
    (4, 3, "and then discuss the Firewood and Water"),
    (5, 6, "would you be willing to compromise on food if I give you more firewood"),
    (5, 8, "2 waters is not possible since you want 1 water"),
    (16, 2, "I get 2 Food, 1 Water, and 1 Firewood, and you get 1 Food, 2 Water, and 1 Firewood"),
    (18, 0, "or we can discuss other options that work for both of us"),
    (18, 8, "and we split the water"),
    (49, 5, "in return, you could give me 1 package of water and 2 packages of food"),
    (49, 11, "or do you have a different distribution in mind"),
    (53, 1, "we split the food and water evenly"),
]


class BidExtractorManualTests(unittest.TestCase):
    def test_curated_manual_positives(self):
        for utterance, expected_self in MANUAL_POSITIVES:
            with self.subTest(utterance=utterance):
                extracted = extract_bid_from_utterance(utterance)
                self.assertIsNotNone(extracted)
                self.assertEqual(extracted["self_counts"], expected_self)
                self.assertEqual(extracted["opp_counts"], _opp_counts(expected_self))

    def test_curated_manual_negatives(self):
        for utterance in MANUAL_NEGATIVES:
            with self.subTest(utterance=utterance):
                self.assertIsNone(extract_bid_from_utterance(utterance))


class BidExtractorProtocol1FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.turns_path = Path("structured_cot/results/protocol1_70b_full/turns.jsonl")
        if not cls.turns_path.exists():
            cls.rows_by_key = None
            return

        rows_by_key = {}
        with cls.turns_path.open() as f:
            for line in f:
                row = json.loads(line)
                rows_by_key[(row.get("dialogue_id"), row.get("turn_index"))] = row
        cls.rows_by_key = rows_by_key

    def _get_row(self, dialogue_id, turn_index):
        if self.rows_by_key is None:
            self.skipTest(f"missing Protocol-1 artifact: {self.turns_path}")
        key = (dialogue_id, turn_index)
        self.assertIn(key, self.rows_by_key, f"missing Protocol-1 row {key}")
        return self.rows_by_key[key]

    def test_protocol1_positive_fixtures_extract_exactly(self):
        for dialogue_id, turn_index, snippet, expected_self in PROTOCOL1_POSITIVES:
            with self.subTest(dialogue_id=dialogue_id, turn_index=turn_index):
                row = self._get_row(dialogue_id, turn_index)
                utterance = (row.get("parsed_utterance") or "").strip()
                self.assertIn(snippet, utterance)
                extracted = extract_bid_from_utterance(utterance)
                self.assertIsNotNone(extracted)
                self.assertEqual(extracted["self_counts"], expected_self)
                self.assertEqual(extracted["opp_counts"], _opp_counts(expected_self))

    def test_protocol1_negative_fixtures_abstain(self):
        for dialogue_id, turn_index, snippet in PROTOCOL1_NEGATIVES:
            with self.subTest(dialogue_id=dialogue_id, turn_index=turn_index):
                row = self._get_row(dialogue_id, turn_index)
                utterance = (row.get("parsed_utterance") or "").strip()
                self.assertIn(snippet, utterance)
                self.assertIsNone(extract_bid_from_utterance(utterance))


if __name__ == "__main__":
    unittest.main()
