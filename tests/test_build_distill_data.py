import unittest

from sft_8b.build_distill_data import is_counter


def turn(role, text):
    return {"id": role, "text": text}


class CounterOfferPredicateTest(unittest.TestCase):
    def test_no_prior_action_is_offer(self):
        history = [
            turn("mturk_agent_2", "hello"),
            turn("mturk_agent_1", "hi"),
        ]
        self.assertFalse(is_counter(history, "mturk_agent_1"))

    def test_opponent_submit_stays_pending_through_natural_text(self):
        history = [
            turn("mturk_agent_2", "Submit-Deal"),
            turn("mturk_agent_2", "this is fair"),
            turn("mturk_agent_1", "let me think"),
        ]
        self.assertTrue(is_counter(history, "mturk_agent_1"))

    def test_own_submit_or_resolving_action_is_not_counter(self):
        own_submit = [
            turn("mturk_agent_1", "Submit-Deal"),
            turn("mturk_agent_2", "please reconsider"),
        ]
        self.assertFalse(is_counter(own_submit, "mturk_agent_1"))

        resolved = [
            turn("mturk_agent_2", "Submit-Deal"),
            turn("mturk_agent_1", "Reject-Deal"),
            turn("mturk_agent_2", "what about this"),
        ]
        self.assertFalse(is_counter(resolved, "mturk_agent_1"))


if __name__ == "__main__":
    unittest.main()

