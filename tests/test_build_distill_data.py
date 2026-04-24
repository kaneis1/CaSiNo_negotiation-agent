import unittest

from sft_8b.build_distill_data import (
    max_prior_opp_submit_age_turns,
    submit_is_response_to_opp_offer,
)


def turn(role, text):
    return {"id": role, "text": text}


class SubmitResponsePredicateTest(unittest.TestCase):
    def test_no_prior_opponent_submit(self):
        history = [
            turn("mturk_agent_2", "hello"),
            turn("mturk_agent_1", "hi"),
        ]
        self.assertFalse(submit_is_response_to_opp_offer(history, "mturk_agent_1"))
        self.assertIsNone(max_prior_opp_submit_age_turns(history, "mturk_agent_1"))

    def test_any_prior_opponent_submit_with_later_text(self):
        history = [
            turn("mturk_agent_2", "Submit-Deal"),
            turn("mturk_agent_2", "this is fair"),
            turn("mturk_agent_1", "let me think"),
        ]
        self.assertTrue(submit_is_response_to_opp_offer(history, "mturk_agent_1"))
        self.assertEqual(max_prior_opp_submit_age_turns(history, "mturk_agent_1"), 2)

    def test_most_recent_prior_opponent_submit_age(self):
        history = [
            turn("mturk_agent_2", "Submit-Deal"),
            turn("mturk_agent_1", "Reject-Deal"),
            turn("mturk_agent_2", "Submit-Deal"),
            turn("mturk_agent_1", "still not quite right"),
        ]
        self.assertTrue(submit_is_response_to_opp_offer(history, "mturk_agent_1"))
        self.assertEqual(max_prior_opp_submit_age_turns(history, "mturk_agent_1"), 1)


if __name__ == "__main__":
    unittest.main()
