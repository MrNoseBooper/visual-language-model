"""Checks for the Left/Right fix-up in common/hands.py.

Run with:  python -m unittest discover tests
"""
import unittest

import numpy as np

from common.hands import Hand, pair_hands, relabel_same_side


def fake_hand(wrist_x, handedness):
    """A hand whose 21 points all sit at wrist_x across the screen (0..1)."""
    points = np.zeros((21, 3), dtype=np.float32)
    points[:, 0] = wrist_x
    return Hand(points, points, handedness, 1.0, 4 / 3)


class RelabelTest(unittest.TestCase):
    def test_both_right_becomes_left_and_right_by_position(self):
        on_right, on_left = fake_hand(0.8, "Right"), fake_hand(0.2, "Right")
        relabel_same_side([on_right, on_left])
        self.assertEqual((on_left.handedness, on_right.handedness), ("Left", "Right"))
        self.assertIs(pair_hands([on_right, on_left])[0], on_left)

    def test_different_labels_are_kept(self):
        # Trust MediaPipe when it gives two different labels, even against screen position.
        a, b = fake_hand(0.8, "Left"), fake_hand(0.2, "Right")
        relabel_same_side([a, b])
        self.assertEqual((a.handedness, b.handedness), ("Left", "Right"))

    def test_one_hand_is_kept(self):
        hand = fake_hand(0.8, "Left")
        relabel_same_side([hand])
        self.assertEqual(hand.handedness, "Left")


if __name__ == "__main__":
    unittest.main()
