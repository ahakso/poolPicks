__author__ = 'Hakso'


import unittest
from scrapeMod import combos
from scrapeMod import probability_favorite_moneyline
import numpy as np
import pandas as pd

class combosTest(unittest.TestCase):
    """Tests for scrapeMod.py"""
    def setUp(self):
        self.all_outcomes, self.odds_instance, self.outcome_odds, self.points_scored = combos(pd.Series([.9, .55, .75]))

    def test_number_of_possible_combinations(self):
        self.assertEqual(self.all_outcomes.shape,(3,8))

    def test_symmetry(self):
        self.assertTrue(np.all(np.equal(self.points_scored,self.points_scored.T)))

    def test_odds_sum_to_one(self):
        self.assertEqual(np.sum(self.outcome_odds),1.0)

class test_probability_favorite_moneyline(unittest.TestCase):

    def test_even_odds(self):
        test_val = probability_favorite_moneyline(110, 110)
        self.assertEqual(test_val,0.5)

    def test_commutivity(self):
        a = -145
        b = 125
        self.assertEqual(probability_favorite_moneyline(a, b),probability_favorite_moneyline(b, a))
if __name__ == '__main__':
    unittest.main()