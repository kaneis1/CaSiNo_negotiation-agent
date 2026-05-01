import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sft_8b.ablation import (
    EvidenceRenderer,
    RuleLikelihoodProvider,
    UniformProvider,
    normalize_posterior,
    parse_probability_response,
)
from sft_8b.ablation_student import (
    AblationStudentTurnAgent,
    parse_student_response_schema,
    posterior_prefix,
)
from sft_8b.posterior import N_ORDERINGS, ORDERINGS


class AblationHarnessTests(unittest.TestCase):
    def test_posterior_providers_return_normalized_six_vectors(self):
        p = UniformProvider().posterior()
        self.assertEqual(len(p), 6)
        self.assertAlmostEqual(float(np.sum(p)), 1.0)

        rule = RuleLikelihoodProvider()
        history = [{"id": "mturk_agent_2", "text": "I really need water for my trip."}]
        p2 = rule.posterior(
            history=history,
            my_role="mturk_agent_1",
            opp_role="mturk_agent_2",
            my_priorities={"High": "Food", "Medium": "Water", "Low": "Firewood"},
            my_reasons={},
        )
        self.assertEqual(len(p2), 6)
        self.assertAlmostEqual(float(np.sum(p2)), 1.0)
        water_high = sum(p2[i] for i, ordering in enumerate(ORDERINGS) if ordering[0] == "Water")
        self.assertGreater(water_high, 1.0 / 3.0)

        offer_p = rule.posterior(
            history=[{
                "id": "mturk_agent_2",
                "text": "Submit-Deal",
                "task_data": {
                    "issue2youget": {"Food": 0, "Water": 3, "Firewood": 0},
                    "issue2theyget": {"Food": 3, "Water": 0, "Firewood": 3},
                },
            }],
            my_role="mturk_agent_1",
            opp_role="mturk_agent_2",
            my_priorities={"High": "Food", "Medium": "Water", "Low": "Firewood"},
            my_reasons={},
        )
        self.assertGreater(
            sum(offer_p[i] for i, ordering in enumerate(ORDERINGS) if ordering[0] == "Water"),
            1.0 / 3.0,
        )

    def test_evidence_renderer_modes(self):
        history = [
            {"id": "mturk_agent_2", "text": "I need water.", "ablation_turn_index": 0, "ablation_strategy_labels": ["self-need"]},
            {"id": "mturk_agent_1", "text": "Hello.", "ablation_turn_index": 1, "ablation_strategy_labels": ["small-talk"]},
            {
                "id": "mturk_agent_2",
                "text": "Submit-Deal",
                "task_data": {
                    "issue2youget": {"Food": 1, "Water": 2, "Firewood": 0},
                    "issue2theyget": {"Food": 2, "Water": 1, "Firewood": 3},
                },
            },
        ]
        utter = EvidenceRenderer(mode="utterance_only").transform_history(
            history, my_role="mturk_agent_1", opp_role="mturk_agent_2", dialogue_id=1
        )
        self.assertEqual([t["text"] for t in utter], ["I need water.", "Hello."])

        plus = EvidenceRenderer(mode="utterance_plus_offers").transform_history(
            history, my_role="mturk_agent_1", opp_role="mturk_agent_2", dialogue_id=1
        )
        self.assertIn("Opponent proposed", plus[-1]["text"])

        offers = EvidenceRenderer(mode="offers_only").transform_history(
            history, my_role="mturk_agent_1", opp_role="mturk_agent_2", dialogue_id=1
        )
        self.assertEqual(len(offers), 1)
        self.assertIn("opponent_share", offers[0]["text"])

        pref = EvidenceRenderer(mode="preference_utterances_only").transform_history(
            history, my_role="mturk_agent_1", opp_role="mturk_agent_2", dialogue_id=1
        )
        self.assertEqual([t["text"] for t in pref], ["I need water."])

        nonpref = EvidenceRenderer(mode="nonpreference_utterances_only").transform_history(
            history, my_role="mturk_agent_1", opp_role="mturk_agent_2", dialogue_id=1
        )
        self.assertEqual([t["text"] for t in nonpref], ["Hello."])

    def test_probability_parser_accepts_posterior_block(self):
        raw = "<posterior>\n" + "\n".join(
            f"p({' > '.join(o)})={1.0 / N_ORDERINGS:.4f}" for o in ORDERINGS
        ) + "\n</posterior>"
        posterior, errors = parse_probability_response(raw)
        self.assertFalse(errors)
        self.assertEqual(len(posterior), 6)
        self.assertAlmostEqual(float(np.sum(posterior)), 1.0)

    def test_student_schema_parsers(self):
        action = """<selected_intent>
utter
</selected_intent>
<selected_content>
null
</selected_content>
<utterance>
I can work with that.
</utterance>"""
        parsed = parse_student_response_schema(action, schema="action_only")
        self.assertIsNone(parsed["parse_error"])
        self.assertIsNone(parsed["posterior"])
        self.assertEqual(parsed["selected_intent"], "utter")

        map_raw = """<posterior>
MAP: Water > Food > Firewood
</posterior>
<selected_intent>
accept
</selected_intent>
<selected_content>
null
</selected_content>
<utterance>

</utterance>"""
        parsed = parse_student_response_schema(map_raw, schema="map_only")
        self.assertIsNone(parsed["parse_error"])
        self.assertEqual(int(np.argmax(parsed["posterior"])), ORDERINGS.index(("Water", "Food", "Firewood")))

        reversed_raw = """<utterance>
Let's split it.
</utterance>
<selected_intent>
submit
</selected_intent>
<selected_content>
{"self_counts":{"Food":1,"Water":2,"Firewood":0},"opp_counts":{"Food":2,"Water":1,"Firewood":3}}
</selected_content>
<posterior>
p(Food > Water > Firewood)=1.0
p(Food > Firewood > Water)=0.0
p(Water > Food > Firewood)=0.0
p(Water > Firewood > Food)=0.0
p(Firewood > Food > Water)=0.0
p(Firewood > Water > Food)=0.0
</posterior>"""
        parsed = parse_student_response_schema(reversed_raw, schema="reversed")
        self.assertIsNone(parsed["parse_error"])
        self.assertEqual(parsed["selected_content"]["self_tuple"], [1, 2, 0])

    def test_prefix_injection_fake_student_model(self):
        class FakeStudent:
            def __init__(self):
                self.prefixes = []
                self.last_raw_response = ""

            def predict(self, **kwargs):
                prefix = kwargs.get("assistant_prefix")
                self.prefixes.append(prefix)
                posterior = [1.0 / 6.0] * 6
                if prefix:
                    posterior, _ = parse_student_response_schema(
                        prefix + "<selected_intent>\nutter\n</selected_intent>\n<selected_content>\nnull\n</selected_content>\n<utterance>\nhello\n</utterance>",
                        schema="full",
                    )["posterior"], None
                return {
                    "posterior": posterior,
                    "selected_intent": "utter",
                    "selected_content": None,
                    "utterance": "hello",
                    "parse_error": None,
                }

        dialogue = {
            "dialogue_id": "d1",
            "participant_info": {
                "mturk_agent_1": {"value2issue": {"High": "Food", "Medium": "Water", "Low": "Firewood"}},
                "mturk_agent_2": {"value2issue": {"High": "Water", "Medium": "Food", "Low": "Firewood"}},
            },
        }
        fake = FakeStudent()
        agent = AblationStudentTurnAgent(
            fake,
            schema="full",
            dialogues_by_id={"d1": dialogue},
            prefix_mode="correct",
            sanity_compare_prefix=True,
        )
        pred = agent.predict_turn(
            history=[],
            my_role="mturk_agent_1",
            opp_role="mturk_agent_2",
            my_priorities=dialogue["participant_info"]["mturk_agent_1"]["value2issue"],
            my_reasons={},
            pending_offer=None,
            dialogue_id="d1",
        )
        self.assertEqual(pred["action"], "utter")
        self.assertEqual(len(fake.prefixes), 2)
        self.assertIsNone(fake.prefixes[0])
        self.assertIn("<posterior>", fake.prefixes[1])
        self.assertEqual(agent.summary["prefix_calls"], 1)


if __name__ == "__main__":
    unittest.main()
