import unittest

from sft_8b.build_day8_sft_data import (
    compute_repeat_map,
    row_to_student_messages,
    stable_eval_split,
)
from sft_8b.student_prompts import extract_tagged_section


class Day8DataTests(unittest.TestCase):
    def test_stable_eval_split_is_deterministic(self):
        split1 = stable_eval_split(156, seed=42, eval_fraction=0.1)
        split2 = stable_eval_split(156, seed=42, eval_fraction=0.1)
        self.assertEqual(split1, split2)

    def test_compute_repeat_map_oversamples_non_utter(self):
        repeat = compute_repeat_map(
            {"utter": 100, "submit": 10, "accept": 5, "reject": 1},
            mode="oversample_to_anchor",
            max_repeat=32,
        )
        self.assertEqual(repeat["utter"], 1)
        self.assertEqual(repeat["submit"], 1)
        self.assertEqual(repeat["accept"], 2)
        self.assertEqual(repeat["reject"], 10)

    def test_row_to_student_messages_builds_tagged_prompt(self):
        row = {
            "style": "balanced",
            "posterior": [0.1, 0.2, 0.1, 0.2, 0.2, 0.2],
            "target": {
                "selected_intent": "submit",
                "selected_content": {
                    "self_counts": {"Food": 1, "Water": 2, "Firewood": 0},
                    "opp_counts": {"Food": 2, "Water": 1, "Firewood": 3},
                    "self_tuple": [1, 2, 0],
                    "opp_tuple": [2, 1, 3],
                },
                "utterance": "Let us split it this way.",
            },
            "messages": [
                {"role": "system", "content": "ignored"},
                {
                    "role": "user",
                    "content": (
                        "<self_priorities>\nHigh: Food\nMedium: Water\nLow: Firewood\n</self_priorities>\n\n"
                        "<self_reasons>\nHigh: need it most\n</self_reasons>\n\n"
                        "<history>\nMe: hi\nOpponent: hello\n</history>\n\n"
                        "<posterior>\np(...)=0.1\n</posterior>\n\n"
                        "<menu>\n...\n</menu>\n\n"
                        "<style>\nbalanced\n</style>"
                    ),
                },
                {"role": "assistant", "content": "{}"},
            ],
        }
        messages = row_to_student_messages(row)
        user = messages[1]["content"]
        assistant = messages[2]["content"]
        self.assertEqual(extract_tagged_section(user, "style_token"), "balanced")
        self.assertIn("Me: hi", extract_tagged_section(user, "history"))
        self.assertEqual(extract_tagged_section(assistant, "selected_intent"), "submit")
        self.assertIn('"self_tuple":[1,2,0]', extract_tagged_section(assistant, "selected_content"))


if __name__ == "__main__":
    unittest.main()
