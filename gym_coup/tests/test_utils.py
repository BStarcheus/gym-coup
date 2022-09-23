import unittest
from gym_coup.utils import *

class TestEncodeObs(unittest.TestCase):
    def test_basic(self):
        # Possible obs at first step of game
        obs = [0, 3, -1, -1, -1, -1, -1, -1,
               0, 0, -1, -1, 0, 0, -1, -1,
               1, 2, -1, -1, 0]
        enc = encode_obs(obs)
        self.assertEqual(len(enc), 123)
        x = [1, 0, 0, 0, 0,
             0, 0, 0, 1, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             1, 0, 1, 0,
             0, 0, 0, 0,
             1, 0, 1, 0,
             0, 0, 0, 0,
             1, 2,
             0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0,
             0]
        self.assertListEqual(list(enc), x)

    def test_face_up_cards(self):
        # Each player has 1 face up card
        obs = [1, 2, -1, -1, 0, -1, -1, -1,
               0, 1, -1, -1, 1, 0, -1, -1,
               0, 4, 2, 7, 1]
        enc = encode_obs(obs)
        self.assertEqual(len(enc), 123)
        x = [0, 1, 0, 0, 0,
             0, 0, 1, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             1, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             1, 0, 0, 1,
             0, 0, 0, 0,
             0, 1, 1, 0,
             0, 0, 0, 0,
             0, 4,
             0, 0, 1, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0,
             1]
        self.assertListEqual(list(enc), x)

    def test_face_up_cards(self):
        # P1 has 4 cards in exchange
        obs = [0, 0, 2, 4, 0, -1, -1, -1,
               0, 0, 0, 0, 1, 0, -1, -1,
               7, 9, 5, 12, 0]
        enc = encode_obs(obs)
        self.assertEqual(len(enc), 123)
        x = [1, 0, 0, 0, 0,
             1, 0, 0, 0, 0,
             0, 0, 1, 0, 0,
             0, 0, 0, 0, 1,
             1, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             1, 0, 1, 0,
             1, 0, 1, 0,
             0, 1, 1, 0,
             0, 0, 0, 0,
             7, 9,
             0, 0, 0, 0, 0, 1, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0,
             0]
        self.assertListEqual(list(enc), x)

    def test_invalid(self):
        # Action 32 invalid
        obs = [0, 0, 2, 4, 0, -1, -1, -1,
               0, 0, 0, 0, 1, 0, -1, -1,
               7, 9, 5, 32, 0]
        with self.assertRaises(IndexError):
            enc = encode_obs(obs)