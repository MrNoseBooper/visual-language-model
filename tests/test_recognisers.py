"""Checks that main.py's menu offers each recogniser only the actions it has.

Run with:  python -m unittest discover tests
"""
import unittest

from recognisers import ACTION_MODULES, actions


class ActionsTest(unittest.TestCase):
    def test_face_tracer_only_runs(self):
        self.assertEqual(actions("faces"), ["run"])

    def test_trainable_recognisers_have_every_action(self):
        for key in ("gestures", "items"):
            self.assertEqual(actions(key), list(ACTION_MODULES))


if __name__ == "__main__":
    unittest.main()
