import tempfile
import unittest
from pathlib import Path

from opponent_model.turn_agents import DistilledStudentTurnAgent
from sft_8b.student_parser import parse_student_response


class _FakeStudentModel:
    def __init__(self, parsed, raw_response=""):
        self._parsed = parsed
        self.last_raw_response = raw_response
        self.base_model = "fake-base"
        self.adapter_path = "fake-adapter"
        self.max_new_tokens = 256
        self.temperature = 0.0
        self.calls = 0

    def predict(self, **kwargs):
        self.calls += 1
        return dict(self._parsed)


class StudentParserTests(unittest.TestCase):
    def test_parse_student_response_valid_submit(self):
        raw = """<posterior>
p(Food > Water > Firewood)=0.1000
p(Food > Firewood > Water)=0.2000
p(Water > Food > Firewood)=0.1000
p(Water > Firewood > Food)=0.2000
p(Firewood > Food > Water)=0.1000
p(Firewood > Water > Food)=0.3000
</posterior>
<selected_intent>
submit
</selected_intent>
<selected_content>
{"self_counts":{"Food":2,"Water":1,"Firewood":3},"opp_counts":{"Food":1,"Water":2,"Firewood":0}}
</selected_content>
<utterance>
Let's lock this in.
</utterance>"""
        parsed = parse_student_response(raw)
        self.assertIsNone(parsed["parse_error"])
        self.assertEqual(parsed["selected_intent"], "submit")
        self.assertEqual(parsed["selected_content"]["self_tuple"], [2, 1, 3])
        self.assertAlmostEqual(sum(parsed["posterior"]), 1.0, places=6)

    def test_parse_student_response_flags_missing_content_for_submit(self):
        raw = """<posterior>
p(Food > Water > Firewood)=1.0
p(Food > Firewood > Water)=0.0
p(Water > Food > Firewood)=0.0
p(Water > Firewood > Food)=0.0
p(Firewood > Food > Water)=0.0
p(Firewood > Water > Food)=0.0
</posterior>
<selected_intent>
submit
</selected_intent>
<selected_content>
null
</selected_content>
<utterance>
</utterance>"""
        parsed = parse_student_response(raw)
        self.assertIsNotNone(parsed["parse_error"])
        self.assertIn("submit intent requires non-null selected_content",
                      parsed["selected_content_errors"])

    def test_parse_student_response_tolerates_truncated_trailing_tag(self):
        raw = """<posterior>
p(Food > Water > Firewood)=0.2500
p(Food > Firewood > Water)=0.2500
p(Water > Food > Firewood)=0.2500
p(Water > Firewood > Food)=0.2500
p(Firewood > Food > Water)=0.0000
p(Firewood > Water > Food)=0.0000
</posterior>
<selected_intent>
utter
</selected_intent>
<selected_content>
null
</selected_content>
<utterance>
I need more water"""
        parsed = parse_student_response(raw)
        self.assertIsNone(parsed["parse_error"])
        self.assertEqual(parsed["selected_intent"], "utter")
        self.assertEqual(parsed["utterance"], "I need more water")

    def test_distilled_student_turn_agent_maps_submit_to_bid(self):
        agent = DistilledStudentTurnAgent(
            _FakeStudentModel({
                "posterior": [0.1, 0.2, 0.1, 0.2, 0.2, 0.2],
                "selected_intent": "submit",
                "selected_content": {
                    "self_counts": {"Food": 1, "Water": 2, "Firewood": 0},
                    "opp_counts": {"Food": 2, "Water": 1, "Firewood": 3},
                    "self_tuple": [1, 2, 0],
                    "opp_tuple": [2, 1, 3],
                },
                "utterance": "I propose this split.",
                "parse_error": None,
            }),
            style="balanced",
        )
        pred = agent.predict_turn(
            history=[],
            my_role="mturk_agent_1",
            opp_role="mturk_agent_2",
            my_priorities={"High": "Food", "Medium": "Water", "Low": "Firewood"},
            my_reasons={},
            pending_offer=None,
        )
        self.assertEqual(pred["action"], "submit")
        self.assertEqual(pred["bid"], {"Food": 1, "Water": 2, "Firewood": 0})
        self.assertIsNone(pred["accept"])

    def test_distilled_student_turn_agent_uses_cache_on_repeat_call(self):
        raw = """<posterior>
p(Food > Water > Firewood)=0.1000
p(Food > Firewood > Water)=0.2000
p(Water > Food > Firewood)=0.1000
p(Water > Firewood > Food)=0.2000
p(Firewood > Food > Water)=0.2000
p(Firewood > Water > Food)=0.2000
</posterior>
<selected_intent>
accept
</selected_intent>
<selected_content>
null
</selected_content>
<utterance>
</utterance>"""
        fake = _FakeStudentModel({
            "posterior": [0.1, 0.2, 0.1, 0.2, 0.2, 0.2],
            "selected_intent": "accept",
            "selected_content": None,
            "utterance": "",
            "parse_error": None,
        }, raw_response=raw)
        with tempfile.TemporaryDirectory() as tmp:
            agent = DistilledStudentTurnAgent(
                fake,
                style="balanced",
                cache_path=Path(tmp) / "student_cache.sqlite",
                parse_log_path=Path(tmp) / "student_parse_failures.jsonl",
            )
            kwargs = dict(
                history=[],
                my_role="mturk_agent_1",
                opp_role="mturk_agent_2",
                my_priorities={"High": "Food", "Medium": "Water", "Low": "Firewood"},
                my_reasons={},
                pending_offer=None,
                dialogue_id=123,
                turn_index=0,
            )
            pred1 = agent.predict_turn(**kwargs)
            pred2 = agent.predict_turn(**kwargs)
            self.assertEqual(fake.calls, 1)
            self.assertEqual(pred1["action"], "accept")
            self.assertEqual(pred2["action"], "accept")
            self.assertEqual(agent.summary["cache_hits"], 1)
            self.assertEqual(agent.summary["cache_misses"], 1)


if __name__ == "__main__":
    unittest.main()
