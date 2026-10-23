"""Checks for the gesture features (recognisers/gestures/features.py).

Run with:  python -m unittest discover tests
"""
import unittest

import numpy as np

from recognisers.gestures.features import NUM_FEATURES, NUM_TWO_HAND_FEATURES, normalise, normalise_two


def fake_hand(seed):
    """21 random points, shaped like Hand.landmarks / Hand.world_landmarks."""
    return np.random.default_rng(seed).random((21, 3)).astype(np.float32)


class NormaliseTest(unittest.TestCase):
    def test_left_hand_matches_mirrored_right_hand(self):
        right = fake_hand(1)
        left = right.copy()
        left[:, 0] *= -1
        np.testing.assert_allclose(normalise(left, "Left"), normalise(right, "Right"), atol=1e-6)

    def test_position_and_size_do_not_matter(self):
        hand = fake_hand(2)
        moved_and_bigger = hand * 2.5 + np.array([1, 2, 3], dtype=np.float32)
        np.testing.assert_allclose(normalise(moved_and_bigger, "Right"), normalise(hand, "Right"), atol=1e-5)

    def test_input_is_not_changed(self):
        hand = fake_hand(3)
        before = hand.copy()
        normalise(hand, "Left")
        normalise_two(hand, hand, hand, hand, 4 / 3)
        np.testing.assert_array_equal(hand, before)

    def test_sizes(self):
        self.assertEqual(normalise(fake_hand(4), "Right").shape, (NUM_FEATURES,))
        two = normalise_two(fake_hand(5), fake_hand(6), fake_hand(7), fake_hand(8), 4 / 3)
        self.assertEqual(two.shape, (NUM_TWO_HAND_FEATURES,))


if __name__ == "__main__":
    unittest.main()
