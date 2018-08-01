import scrapeMod as sm
import pandas as pd
import numpy as np
import time
from importlib import reload

week_number = 8
year = 2017
"""Get data for current week"""
try:
    df_moneylines = pd.read_csv('csv/weekMoneylines{}_{}.csv'.format(week_number, year), index_col=0)
except FileNotFoundError:
    sm.moneyline_csv(week_number, True)
    print('Created the csv file')
n_games = df_moneylines.shape[0]
all_outcomes, points_scored = sm.combos(n_games)
outcome_odds = sm.all_outcome_odds(df_moneylines['favorite_odds'], all_outcomes)

"""Generate predicted picks"""
# Build player models based on picks and odds csv files from previous weeks
clf, players, variable_names = sm.opponent_models_random_forest()
# Generate feature frame
X = sm.engineered_features(df_moneylines)
# Predict Opponent Picks
picks_pred = sm.generate_picks(clf, X)

"""Get the results based on predicted picks"""
# Get the value of the top 500 picks based on predicted opponent picks
total_value_pick_pred, max_score_pred, opponent_pick_idx_pred, _ = sm.calculate_pick_values(
    all_outcomes, points_scored, outcome_odds, picks_pred, df_moneylines)
ind = sm.best_pick_indices(total_value_pick_pred)
best_seattle_pick_idx = sm.best_seattle_pick(ind, df_moneylines, all_outcomes)

"""View results"""
results_best_pick = sm.get_pick_odds(ind[0], points_scored, max_score_pred, outcome_odds, total_value_pick_pred,
                                     picks_pred.shape[0], all_outcomes)
results_best_SEA = sm.get_pick_odds(best_seattle_pick_idx, points_scored, max_score_pred, outcome_odds,
                                    total_value_pick_pred, picks_pred.shape[0], all_outcomes)
if best_seattle_pick_idx != ind[0]:
    print('Picking the Hawks brings roi from {:.1f}% to {:.1f}%'.format(results_best_pick['roi'],results_best_SEA['roi']))
    print('Seattle picks:')
    sm.binary_picks_to_team_names(all_outcomes[:, best_seattle_pick_idx], df_moneylines)
print('Best picks:')
sm.binary_picks_to_team_names(all_outcomes[:, ind[0]], df_moneylines)


"""Once the actual picks are in"""
# Read in the picks that were made
"""Get data for current week"""
fn_picks = 'csv/picks{}_{}.csv'.format(week_number, year)
try:
    picks = pd.read_csv(fn_picks, index_col=0)
except FileNotFoundError:
    sm.pick_csv(week_number)
    picks = pd.read_csv(fn_picks, index_col=0)
    print('Created the csv file')

my_picks = pd.read_csv('csv/mypicks{}_{}.csv'.format(week_number, year), index_col=0)
picks_made = picks.loc[np.sum(picks.isnull(),axis=1)<5,:] #removes people who didn't get picks in

total_value_pick, max_score, opponent_pick_idx, my_pick = sm.calculate_pick_values(
    all_outcomes, points_scored, outcome_odds, picks, df_moneylines)
ind = sm.best_pick_indices(total_value_pick)
results_final = sm.get_pick_odds(ind[0], points_scored, max_score, outcome_odds, total_value_pick, picks_made.shape[0],
                                 all_outcomes)
results_my_pick = sm.get_pick_odds(my_pick, points_scored, max_score, outcome_odds, total_value_pick,
                                   picks_made.shape[0], all_outcomes)
print('{:16s} {:.1f}\n{:16s} {:.1f}\n{:16s} {:.1f}'.format(
    'Predicted roi', results_best_SEA['roi'], 'Best roi:', results_final['roi'], 'Actual roi:', results_my_pick['roi']))
