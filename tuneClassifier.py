import scrapeMod as sm
import pandas as pd
import numpy as np
import sklearn
from importlib import reload


def loocv_picks(week_excluded,df_moneylines):
    """Build player models based on picks and odds csv files from previous weeks"""
    clf, players, variable_names = sm.opponent_models_random_forest(week_excluded)
    #clf, players, variable_names = sm.opponent_models_svm(week_excluded)
    #clf, players, variable_names = sm.opponent_models_kneighbors(week_excluded)
    # Generate feature frame
    X = sm.engineered_features(df_moneylines)
    # Predict Opponent Picks
    picks_pred = sm.generate_picks(clf, X)
    return picks_pred, clf, variable_names


n_picks = 0
n_correct = 0
for week_excluded in range(1,16):#range(16):
    df_moneylines = pd.read_csv('weekMoneylines' + str(week_excluded) + '.csv', index_col=0)
    """Generate predicted picks"""
    picks_pred, clf, variable_names = loocv_picks(week_excluded, df_moneylines)
    if isinstance(clf[0],sklearn.ensemble.forest.RandomForestClassifier):
        values = sm.variable_importance(clf,variable_names)

    """Get the actual picks"""
    picks = pd.read_csv('picks'+str(week_excluded) + '.csv',index_col = 0)
    picks_made = picks.loc[np.sum(picks.isnull(),axis=1)<5,:] #removes people who didn't get picks in
    picks_made_binary = sm.picks_names2binary(picks_made, df_moneylines['favorite'])

    """Compare the results"""
    n_picks_week, n_correct_week = sm.evaluate_prediction(picks_pred,picks_made_binary)
    n_picks += n_picks_week
    n_correct += n_correct_week
n_correct/n_picks
