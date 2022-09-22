import unittest
import gym
from gym_coup.envs.coup_env import *

class TestCoupEnvBase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = gym.make('coup-v0')

    def setUp(self):
        self.env.reset()

class TestCoupEnv(TestCoupEnvBase):
    def test_reset_obs(self):
        # Reset called in setup

        obs = self.env.get_obs()
        self.assertIsInstance(obs, tuple)
        self.assertEqual(len(obs), 21)

        # P1 cards are dealt
        for i in range(2):
            self.assertGreaterEqual(obs[i], 0)
            self.assertLessEqual(obs[i], 4)

        # Other cards are hidden or nonexistent
        for i in range(2, 8):
            self.assertEqual(obs[i], -1)

        # 2 cards face down per player
        for i in [8, 9, 12, 13]:
            self.assertEqual(obs[i], 0)
        # Other cards nonexistent
        for i in [10, 11, 14, 15]:
            self.assertEqual(obs[i], -1)

        # P1 1 coin
        self.assertEqual(obs[16], 1)
        # P2 2 coins
        self.assertEqual(obs[17], 2)

        # No last action
        for i in range(18, 20):
            self.assertEqual(obs[i], -1)

        # P1 first turn
        self.assertEqual(obs[20], 0)

        # P2 cards are dealt
        # Get obs from other perspective so not hidden
        obs = self.env.get_obs(p2_view=True)
        for i in range(2):
            self.assertGreaterEqual(obs[i], 0)
            self.assertLessEqual(obs[i], 4)


class TestGeneralActions(TestCoupEnvBase):
    def test_income(self):
        self.env.step(INCOME)
        self.env.step(INCOME) # Go back to P1 turn
        obs, r, term, _ = self.env.last()
        self.assertEqual(obs[16], 2)
        self.assertEqual(r, 0)
        self.assertEqual(term, False)

    def test_pass_foreign_aid(self):
        self.env.step(FOREIGN_AID)
        self.assertListEqual(self.env.get_valid_actions(), [PASS_FA, BLOCK_FA])
        self.env.step(PASS_FA)
        self.env.step(INCOME) # Go back to P1 turn
        obs, r, term, _ = self.env.last()
        self.assertEqual(obs[16], 3)
        self.assertEqual(r, 0)
        self.assertEqual(term, False)

    def test_block_foreign_aid(self):
        self.env.step(FOREIGN_AID)
        self.env.step(BLOCK_FA)
        self.assertListEqual(self.env.get_valid_actions(), [PASS_FA_BLOCK, CHALLENGE_FA_BLOCK])
        self.env.step(PASS_FA_BLOCK)
        self.env.step(INCOME) # Go back to P1 turn
        obs, r, term, _ = self.env.last()
        self.assertEqual(obs[16], 1)
        self.assertEqual(r, 0)
        self.assertEqual(term, False)

    def test_challenge_foreign_aid_block_s(self):
        # Test a challenge succeeding (opp doesn't have Duke)
        while DUKE in self.env.get_obs(p2_view=True)[0:2]:
            self.env.reset()

        self.env.step(FOREIGN_AID)
        self.env.step(BLOCK_FA) # P2 doesn't have Duke
        self.env.step(CHALLENGE_FA_BLOCK)
        obs, _, _, _ = self.env.last()
        self.assertEqual(obs[-1], 1) # It's P2's action (lost challenge)
        self.assertListEqual(self.env.get_valid_actions(), [LOSE_CARD_1, LOSE_CARD_2])

    def test_coup_lose_card(self):
        for i in range(11):
            self.env.step(INCOME)
        self.env.step(COUP)
        self.env.step(LOSE_CARD_1)
        obs, r, term, _ = self.env.last()
        self.assertEqual(obs[-1], 0) # It's P1's action
        self.assertEqual(obs[8], 1) # face up
        self.assertEqual(obs[17], 0) # P2 coins
        self.assertEqual(r, -1)

        self.env.step(INCOME) # Go back to P2 turn
        _, r, _, _ = self.env.last()
        self.assertEqual(r, 1)


def deal_card(env, card):
    while card not in env.get_obs()[0:2]:
        env.reset()


class TestAssassin(TestCoupEnvBase):
    def setUp(self):
        super().setUp()
        deal_card(self.env, ASSASSIN)

    def test_assassinate(self):
        self.env.step(FOREIGN_AID)
        self.env.step(PASS_FA)
        self.env.step(INCOME) # Go back to P1 turn
        self.env.step(ASSASSINATE)
        self.assertListEqual(self.env.get_valid_actions(), [LOSE_CARD_1, LOSE_CARD_2, BLOCK_ASSASSINATE, CHALLENGE_ASSASSINATE])
        self.env.step(LOSE_CARD_1)
        obs, r, term, _ = self.env.last()
        self.assertEqual(obs[-1], 1) # It's P2's action
        self.assertEqual(obs[8], 1) # face up
        self.assertEqual(obs[17], 0) # P1 coins
        self.assertEqual(r, -1)

        self.env.step(INCOME) # Go back to P1 turn
        _, r, _, _ = self.env.last()
        self.assertEqual(r, 1)

    def test_double_assassinate(self):
        self.env.step(FOREIGN_AID)
        self.env.step(PASS_FA)
        self.env.step(INCOME) # Go back to P1 turn
        self.env.step(ASSASSINATE)
        self.env.step(CHALLENGE_ASSASSINATE)
        _, r, term, _ = self.env.last()
        self.assertEqual(r, -2)
        self.assertEqual(term, True)


class TestAmbassador(TestCoupEnvBase):
    def setUp(self):
        super().setUp()
        deal_card(self.env, AMBASSADOR)

    def test_exchange(self):
        self.env.step(EXCHANGE)
        self.assertListEqual(self.env.get_valid_actions(), [PASS_EXCHANGE, CHALLENGE_EXCHANGE])
        self.env.step(PASS_EXCHANGE)
        self.assertListEqual(self.env.get_valid_actions(), [EXCHANGE_RETURN_34, EXCHANGE_RETURN_13,
                                                            EXCHANGE_RETURN_14, EXCHANGE_RETURN_23,
                                                            EXCHANGE_RETURN_24, EXCHANGE_RETURN_12])
        self.env.step(EXCHANGE_RETURN_12)
        _, r, term, _ = self.env.last()
        self.assertEqual(r, 0)
        self.assertEqual(term, False)


class TestCaptain(TestCoupEnvBase):
    def setUp(self):
        super().setUp()
        deal_card(self.env, CAPTAIN)

    def test_steal(self):
        self.env.step(STEAL)
        self.assertListEqual(self.env.get_valid_actions(), [PASS_STEAL, BLOCK_STEAL, CHALLENGE_STEAL])
        self.env.step(PASS_STEAL)
        obs, _, _, _ = self.env.last()
        self.assertEqual(obs[16], 0)
        self.env.step(INCOME) # Go back to P1 turn
        obs, r, term, _ = self.env.last()
        self.assertEqual(obs[16], 3)
        self.assertEqual(r, 0)
        self.assertEqual(term, False)

    def test_block_steal(self):
        self.env.step(INCOME)
        self.env.step(STEAL)
        self.env.step(BLOCK_STEAL)
        self.assertListEqual(self.env.get_valid_actions(), [PASS_STEAL_BLOCK, CHALLENGE_STEAL_BLOCK])
        self.env.step(CHALLENGE_STEAL_BLOCK)
        obs, r, term, _ = self.env.last()
        self.assertEqual(obs[-1], 1) # It's P2's action (lost challenge)
        self.assertListEqual(self.env.get_valid_actions(), [LOSE_CARD_1, LOSE_CARD_2])
        self.assertEqual(obs[17], 2) # P1 block succeeds, keeps their coins
        self.assertEqual(r, 0)
        self.assertEqual(term, False)


class TestContessa(TestCoupEnvBase):
    def setUp(self):
        super().setUp()
        deal_card(self.env, CONTESSA)

    def test_block_assassinate(self):
        self.env.step(INCOME)
        self.env.step(INCOME)
        self.env.step(INCOME)
        self.env.step(ASSASSINATE)
        self.env.step(BLOCK_ASSASSINATE)
        self.assertListEqual(self.env.get_valid_actions(), [PASS_ASSASSINATE_BLOCK, CHALLENGE_ASSASSINATE_BLOCK])
        self.env.step(CHALLENGE_ASSASSINATE_BLOCK)
        obs, r, term, _ = self.env.last()
        self.assertEqual(obs[-1], 1) # It's P2's action (lost challenge)
        self.assertListEqual(self.env.get_valid_actions(), [LOSE_CARD_1, LOSE_CARD_2])
        self.assertEqual(obs[16], 0)


class TestDuke(TestCoupEnvBase):
    def setUp(self):
        super().setUp()
        deal_card(self.env, DUKE)

    def test_challenge_foreign_aid_block_f(self):
        self.env.step(INCOME)
        self.env.step(FOREIGN_AID)
        self.env.step(BLOCK_FA) # P1 has Duke
        self.env.step(CHALLENGE_FA_BLOCK)
        obs, _, _, _ = self.env.last()
        self.assertEqual(obs[-1], 1) # It's P2's action (lost challenge)
        self.assertListEqual(self.env.get_valid_actions(), [LOSE_CARD_1, LOSE_CARD_2])

    def test_tax(self):
        self.env.step(TAX)
        self.assertListEqual(self.env.get_valid_actions(), [PASS_TAX, CHALLENGE_TAX])
        self.env.step(PASS_TAX)
        self.env.step(INCOME) # Go back to P1 turn
        obs, r, term, _ = self.env.last()
        self.assertEqual(obs[16], 4)
        self.assertEqual(r, 0)
        self.assertEqual(term, False)
