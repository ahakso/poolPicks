# Functions that scrape nfl line data from espn and assist in selecting best picks
# given nGames NFL games, return an array with the probability of each game in each scenario, the probability of each
# scenario, and the points scored for each possible prediction for each possible scenario

import itertools
import numpy as np
from urllib.request import urlopen as u_req
import urllib
from bs4 import BeautifulSoup as Soup
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import h5py
import pdb
import scipy
# import time
# from sklearn.model_selection import train_test_split
# import sys
# import cProfile


class MissingGame(ValueError):
    pass


def all_outcome_odds(favorite_odds, all_outcomes):
    """Run primary calculations on possible results and associated odds

    Keyword argument:
    favorite_odds -- the probability that the favorite will win a given game, as output by getWeekResults or other
                     function that calculates from moneyline
    all_outcomes -- binary matrix of all possible outcomes, of dim[n_games x nPossible outcomes]. True = favorite wins

    Output arguments:
    odds_instance -- same shape as all_outcomes, gives odds for individual game outcomes
    outcome_odds -- odds of each outcome
    """

    # Tile odds of favorite winning to match dimensions of all possibilities
    n_outcomes = all_outcomes.shape[1]
    favorite_odds_m = np.tile(favorite_odds[:, None], (1, n_outcomes))

    # Generate the probabilities of each outcomes in all outcomes matrix
    # [n_games x nPossible outcomes]
    odds_instance = np.zeros(favorite_odds_m.shape)
    odds_instance[all_outcomes] = favorite_odds_m[all_outcomes]
    odds_instance[np.logical_not(all_outcomes)] = 1 - favorite_odds_m[np.logical_not(all_outcomes)]

    # Get the probability of each scenario
    outcome_odds = np.prod(odds_instance, 0)
    return outcome_odds


def best_pick_indices(total_value_pick):
    """ Get the indices of the best and top 500 best picks, as indexed in the column of all_outcomes"""
    # Best Pick
    ind = np.argpartition(total_value_pick, -500)[-500:]
    ind = ind[np.argsort(-total_value_pick[ind])]
    return ind


def best_seattle_pick(ind, df, all_outcomes):
    """ Identify the column in all_outcomes that returns the most optimal pick that includes the hawks winning"""
    seattle_row = np.nonzero(np.array('SEA' == df[['home', 'away']]))[0][0]
    seattle_favorite = bool((df['favorite'] == 'SEA').sum())
    seattle_pick_indices = np.nonzero(all_outcomes[seattle_row, ind] == seattle_favorite)[0]
    if len(seattle_pick_indices) == 0:
        print('None of the top {} picks include Seattle winning... tread lightly'.format(len(ind)))
    best_seattle_pick_idx = ind[seattle_pick_indices[0]]
    if best_seattle_pick_idx == ind[0]:
        print('Rest easy, the best pick calls for a Hawks win!')
    return best_seattle_pick_idx


def binary_picks_to_team_names(picks_to_make, df):
    """ Get the team names of the teams to pick

    :param picks_to_make: a binary series where True indicates pick favorite, False indicates pick underdog
    :param df: the dataframe including columns of home, away, favorite
    :return: the names of the teams to pick
    """
    df['underdog'] = df['away'][df['home'] == df['favorite']]
    df['underdog'][df['home'] != df['favorite']] = df['home'][df['home'] != df['favorite']]
    df['picks_to_make'] = df['favorite'][picks_to_make]
    df['picks_to_make'][np.logical_not(picks_to_make)] = df['underdog'][np.logical_not(picks_to_make)]
    print(df['picks_to_make'])


def build_player_training_frames(player_name, week_start=1, week_end=17, week_excluded=None, year=2017):
    """Read in all pick and game data to produce a dataframe of features and outcomes to train model for one player

    Keyword arguments:
    player_name -- the name of the player as indexed in the picks dataframe and on the cbssports website
    week_start -- the first week to use, defaults to 1
    week_end -- the last week to use, default to all available weeks
    week_excluded -- week to leave out in training features, useful for LOOCV

    Output arguments:
    X -- input dataframe of features
    y -- Binary series indicating whether or not the player picked the favorite
    """
    df_total = pd.DataFrame()
    file_not_found_error = (None, None, None)
    for week_number in [x for x in range(week_start, week_end+1) if x != week_excluded]:
        fn_pick = 'csv/picks{}_{}.csv'.format(week_number, year)
        fn_ml = 'csv/weekMoneylines{}_{}.csv'.format(week_number, year)
        try:
            picks = pd.read_csv(fn_pick, index_col=0)
            file_not_found_error = (False, week_number, fn_pick)
        except FileNotFoundError:
            if df_total.shape[0] > 0:
                file_not_found_error = (True, week_number, None)
            else:
                file_not_found_error = (True, week_number, fn_pick)
            return engineered_features(df_total), df_total['picked_favorite'], file_not_found_error
        picks_made = picks.loc[np.sum(picks.isnull(), axis=1) < 5, :]  # removes people who didn't get picks in
        df_moneylines = pd.read_csv(fn_ml, index_col=0)
        df_moneylines = validate_and_align_input_data(df_moneylines, picks_made)
        df = df_moneylines.loc[:, ['season', 'Week', 'home', 'away', 'favorite', 'favorite_odds']]
        temp = pd.DataFrame(picks.loc[player_name].isin(df.favorite))
        temp.index = df.index
        df['picked_favorite'] = temp
        df['week_competitiveness'] = df['favorite_odds'].sum() / df.shape[0]
        df_total = df_total.append(df, ignore_index=False)
    y = df_total['picked_favorite']
    x = engineered_features(df_total)
    return x, y, file_not_found_error


def calculate_pick_values(all_outcomes, points_scored, outcome_odds, picks, favorite_team, my_picks=None):
    """Calculate the value of all picks, knowing the opponent picks and the odds

    Keyword arguments:
    all_outcomes -- all possible combinations of outcomes
    points_scored -- points scored with each possible prediciton for each possible outcome
    outcome_odds -- Odds of each particular outcome realizing
    picks_made -- picks made by the opponents
    df_moneylines -- odds of favorite winning each game
    """

    # Find the index of the opponent pick
    opponent_pick_idx = get_pick_indices(picks, favorite_team, all_outcomes)
    if my_picks is not None:
        my_pick_idx = get_pick_indices(my_picks, favorite_team, all_outcomes)
    else:
        my_pick_idx = None

    # max_score(j) = best opponent score for outcome j
    max_score, tie_value = pick_scores(opponent_pick_idx, points_scored)

    # total_value = the sum of the outright wins and the ties, with tie values accounting for the number of tied players
    total_value = np.greater(points_scored, max_score) + np.multiply(np.equal(points_scored, max_score), tie_value)

    # total_value_pick(i) = total_value of the pick i (integral over all outcomes)
    total_value_pick = np.sum(np.multiply(total_value, outcome_odds), axis=1)

    return total_value_pick, max_score, opponent_pick_idx, my_pick_idx


def combos(n_games):
    """Get objects representing all possible outcomes and associated points scored

    Keyword argument -- n_games

    Output arguments:
    all_outcomes -- binary matrix of all possible outcomes, of dim [n_games x nPossible outcomes]. True = favorite wins
    points_scored -- points scored for each prediction possibility for each outcome
    """
    # [n_games x nPossible outcomes] = [n_games x 2^n_games] array with bool values, True if favorite wins
    all_outcomes = np.asarray([list(i) for i in itertools.product([True, False], repeat=n_games)]).T
    n_outcomes = all_outcomes.shape[1]
    # points_scored(i,j) = points scored for prediction i in outcome j
    try:
        f = h5py.File('cache/points_scored.h5', 'r')
        points_scored = f[str(n_games)]
    except OSError:
        points_scored = np.zeros((n_outcomes, n_outcomes), np.int8)

        # Calculate the number of points score for each guess and each outcome
        for i in range(n_outcomes):
            particular_outcome = all_outcomes[:, i]
            points_scored[0:i, i] = np.sum(np.equal(all_outcomes[:, 0:i], particular_outcome[:, None]), 0)
        points_scored = points_scored+points_scored.T
        np.fill_diagonal(points_scored, n_games)
    return all_outcomes, points_scored


def engineered_features(df):
    """convert the dataframe passed in, which contains largely raw data, and insert derived features

    Keyword arguments:
    df -- dataframe containing, at a minimum, columns ['season', 'Week', 'home', 'away', 'favorite', 'favorite_odds']

    Output arguments:
    X -- dataframe with features for prediction
    """
    def ftr_games_in_week(df):
        df = df.assign(games_in_week=np.nan)
        for i_week in df.Week.unique():
            df.loc[df.Week == i_week, 'games_in_week'] = int(np.sum(df.Week == i_week))
        return df

    def ftr_week_competitiveness(df):
        """Determine mean odds of game if df argument is a single week of games"""
        if (df.shape[0] > 10) & (df.shape[0] < 17):
            df['week_competitiveness'] = df['favorite_odds'].sum() / df.shape[0]
        return df

    def ftr_categoricals(df):
        home_favorite_bool = df.favorite == df.home
        df = df.assign(underdog=df.away[home_favorite_bool])
        df.loc[np.logical_not(home_favorite_bool), 'underdog'] = df.home[np.logical_not(home_favorite_bool)]
        all_teams = ['NYG', 'ATL', 'DAL', 'LAR', 'SEA', 'LAC', 'OAK', 'BAL', 'CAR', 'ARI', 'TEN', 'KC', 'NE', 'PIT',
                     'TB', 'CIN', 'DET', 'GB', 'DEN', 'WAS', 'SF', 'MIA', 'NYJ', 'CLE', 'BUF', 'IND', 'JAC', 'PHI',
                     'NO', 'MIN', 'CHI', 'HOU']
        fav_categorical = df['favorite'].astype('category', categories=all_teams)
        udr_categorical = df['underdog'].astype('category', categories=all_teams)
        favorite_dummies = pd.get_dummies(fav_categorical, prefix='fav')
        underdog_dummies = pd.get_dummies(udr_categorical, prefix='udr')
        df = df.drop('underdog', axis=1)
        df = df.drop('favorite', axis=1)
        df = pd.concat([df, favorite_dummies, underdog_dummies], axis=1)
        return df

    # In the event that a dataframe is passed in with additional columns, get rid of the excess (e.g. moneylines)
    df = df.loc[:, ['season', 'Week', 'home', 'away', 'favorite', 'favorite_odds', 'week_competitiveness']]
    home_favorite_bool = df.favorite == df.home
    df['home_favorite'] = home_favorite_bool
    # df['TX_favorite'] = df['favorite'] == 'HOU'
    # df['TX_underdog'] = ((df['away'] == 'HOU') | (df['home'] == 'HOU')) &  df = ftr_categoricals(df)
    df = ftr_week_competitiveness(df)
    df = ftr_games_in_week(df)
    df = ftr_categoricals(df)
    # x = df[['favorite_odds', 'home_favorite', 'week_competitiveness', 'games_in_week', 'TX_favorite', 'TX_underdog']]
    l1 = ['favorite_odds', 'home_favorite', 'week_competitiveness', 'games_in_week']
    l2 = df.filter(regex='udr_*').columns.tolist()
    l3 = df.filter(regex='fav_[A-Z]{2,3}').columns.tolist()
    x = df[l1+l2+l3]
    # x = df[['favorite_odds', 'home_favorite', 'week_competitiveness', 'games_in_week', 'favorite', 'underdog']]
    return x


def evaluate_prediction(picks_pred, picks_made_binary):
    # Adjust predicted picks by ignoring picks that weren't made and sorting to match
    picks_pred.drop(picks_pred.index.difference(picks_made_binary.index))
    picks_pred = picks_pred.reindex(columns=picks_made_binary.columns, index=picks_made_binary.index)
    match = pd.DataFrame(picks_pred == picks_made_binary)
    n_picks = match.size
    n_correct = match.values.sum()
    return n_picks, n_correct


def generate_picks(clf, x):
        """Generate pool picks based on classifiers for each player

        Keyword arguments:
        clf -- a list of classifiers, one for each player
        X -- the input feature dataframe the classifier will use to predict outcomes
        """
        n_players = len(clf)
        y_pred = []
        players = get_opponent_names()
        for i_player in range(n_players):
            y_pred.append(clf[i_player].predict(x))
        picks_pred = pd.DataFrame(y_pred, index=players, columns=x.index)
        return picks_pred


def get_opponent_names():
    week_number = 12
    year = 2017
    picks = pd.read_csv('csv/picks{}_{}.csv'.format(week_number, year), index_col=0)
    players = list(picks.index)
    return players


def get_pick_indices(picks_made, favorite, all_outcomes):
    """
    Return the column index of the pick in the binary matrix showing all possible outcomes

    :param picks_made: opponent picks (can either be names of teams or binary fav/not favorite
    :param favorite: favorite team names
    :param all_outcomes: binary matrix of all possible outcomes
    :return:
    """
    n_opponents = picks_made.shape[0]
    if type(picks_made.iloc[0, 0]) is np.bool_:
        picks_binary = np.array(picks_made)
    else:
        picks_binary = np.array(picks_names2binary(picks_made, favorite))
    pick_idx = np.zeros(n_opponents, dtype=np.int64)
    for iOpponent in range(n_opponents):
        pick_idx[iOpponent] = np.where(np.all(np.equal(all_outcomes, picks_binary[iOpponent, :, None]), axis=0))[0][0]
    if len(pick_idx) == 1:
        pick_idx = pick_idx[0]
    return pick_idx


def get_odds_current_week(week_number=None):
    """Get odds for the current week

    Output parameters:
    df -- dataFrame with columns (home/away is favorite, odds in favor of favorite, favorite team name)
    """
    # Define functions for cleaner code
    def get_favorite_team_names(game_names, n_games, favorite):
        favorite_team_name_full = []
        home = []
        away = []
        for g in range(n_games):
            s = game_names[g].split()
            idx_at = s.index('at')
            idx_dash = s.index('-')
            if favorite[g] == 'Home':
                favored_team = " ".join(s[(idx_at + 1):idx_dash])
                # under_dog_team = " ".join(s[0:idx_at])
            else:
                favored_team = " ".join(s[0:idx_at])
                # under_dog_team = " ".join(s[(idx_at + 1):idx_dash])
            favorite_team_name_full.append(team_name2abbrv(favored_team)[0])
            home_current_game = team_name2abbrv(' '.join(s[(idx_at + 1):idx_dash]))[0]
            away_current_game = team_name2abbrv(' '.join(s[0:idx_at]))[0]
            home.append(home_current_game)
            away.append(away_current_game)

        return favorite_team_name_full, home, away

    def add_game_results(spread, favorite, spread_p, moneyline_p):
        if np.mean(spread) > 0:
            favorite.append('Home')
        else:
            favorite.append('Away')
        spread_p = np.append(spread_p, np.mean(spreads_p_current_game))
        moneyline_p = np.append(moneyline_p, np.mean(moneylines_p_current_game))
        return spread_p, moneyline_p, favorite

    def reset_one_game_variables():
        spread = np.array([])
        temp_spreads_p = np.array([])
        temp_moneylines_p = np.array([])
        return spread, temp_spreads_p, temp_moneylines_p

    def parse_current_source(current_row):
        row_data = current_row.find_all("td")
        if row_data[1].td is not None:
            spread_current_source = float(row_data[1].td.contents[0])
        else:
            spread_current_source = None
        if len(row_data) < 5:
            moneyline_current_source = [None, None]
        else:
            moneyline_current_source = \
                [re.findall('-?\d+', line) for line in [row_data[7].td.contents[i] for i in [0, 2]]]
            moneyline_current_source = [int(moneyline_current_source[0][0]), int(moneyline_current_source[1][0])]
        return spread_current_source, moneyline_current_source

    if week_number is None:
        week_number = int(input('What is the current week?\n'))
    # ## Get the page
    my_url = 'http://www.espn.com/nfl/lines'

    # open connection
    class NoInternet(Exception):
        pass
    try:
        u_client = u_req(my_url)
    except urllib.error.URLError:
        raise NoInternet("\n\nNot connected to the internet")

    # download page
    page_html = u_client.read()
    u_client.close()

    # let beautiful soup parse it
    page_soup = Soup(page_html, "html.parser")

    # Extract the table
    page_table = page_soup.table

    # Get the games
    games = page_table.find_all('tr', {'class': 'stathead'})
    n_games = len(games)
    game_names = [g.string[0:-5] for g in games]

    # ## Calculate probabilities from moneyline and spread

    # ### Calculate probabilities
    # #### 1) Get moneyline and spread
    # #### 2) convert each valid one to a probability
    # #### 3) Average probabilities

    moneylines_p_current_game = np.array([])  # Hold the probability based on moneyline for each source
    spreads_p_current_game = np.array([])  # Hold the probability based on spread for each source
    spreads_current_game = np.array([])  # Hold the spread for each source for a game

    moneyline_p = np.array([])  # Hold the probability based on moneyline for each game
    spread_p = np.array([])  # Hold the probability based on spread for each game

    # Home/away favorite?
    favorite = []
    # Iterate through all rows in table, includes spreads, moneylines, team names, footers etc
    for row in games[0].next_siblings:
        is_new_game = row["class"] == ['stathead']  # Contains info on the game (e.g. teams)
        is_new_source = ((row["class"] == ['oddrow']) or (row["class"] == ['evenrow'])) and (row.p is None)
        if is_new_game:  # process information from just finished game
            spread_p, moneyline_p, favorite = add_game_results(spreads_current_game, favorite, spread_p, moneyline_p)
            spreads_current_game, spreads_p_current_game, moneylines_p_current_game = reset_one_game_variables()
        if is_new_source:
            spread_current_source, moneyline_current_source = parse_current_source(row)
            if spread_current_source is not None:
                # add this source to array holding spreads
                spreads_current_game = np.append(spreads_current_game, spread_current_source)
            else:
                continue
            # Get probabilities from spread and moneyline (if available)
            spreads_p_current_game = np.append(spreads_p_current_game,
                                               probability_favorite_spread(spread_current_source))
            if any([line == 0 for line in moneyline_current_source]):
                continue
            else:
                moneylines_p_current_game = np.append(moneylines_p_current_game,
                                                      probability_favorite_moneyline(moneyline_current_source[0],
                                                                                     moneyline_current_source[1]))
    spread_p, moneyline_p, favorite = add_game_results(spreads_current_game, favorite, spread_p, moneyline_p)
    favorite_team_name, home, away = get_favorite_team_names(game_names, n_games, favorite)
    game_colname = []
    for game in range(int(len(home))):
        game_colname.append(away[game]+'_'+home[game])
    df = pd.DataFrame(columns=['season', 'Week', 'home', 'away', 'favorite', 'favorite_odds'], index=game_colname)
    team_name2abbrv(favorite_team_name)
    df['home'] = home
    df['away'] = away
    df['favorite'] = favorite_team_name
    # df['favorite home or away'] = favorite
    df['favorite_odds'] = moneyline_p
    df['season'] = 2017
    df['Week'] = week_number
    return df


def get_pick_odds(pick_index, points_scored, max_score, outcome_odds, total_value_pick, n_opponents, all_outcomes):
    idx_winning_outcome = np.greater(points_scored[pick_index, :], max_score)
    idx_tying_outcome = np.equal(points_scored[pick_index, :], max_score)
    results = {
        'n_winning_outcomes': sum(idx_winning_outcome),
        'n_tying_outcomes': sum(idx_tying_outcome),
        'winning_odds': sum(outcome_odds[idx_winning_outcome]),
        'tying_odds': sum(outcome_odds[idx_tying_outcome]),
        'total_value_this_pick': total_value_pick[pick_index],
        'normalized': total_value_pick[pick_index]/(1./n_opponents),
        'roi': 100*(total_value_pick[pick_index]/(1./n_opponents)-1),
        'n_underdog_picks': np.logical_not(all_outcomes[:, pick_index]).sum()
    }
    return results


def get_scores_for_prediction(idx, points_scored):
    """produce an array of scores for each possible outcome

    Keyword arguments:
    idx -- outcome index of the opponent pick. 1d array
    points_scored -- 2D array of number of points scored for each possible selection and outcome

    Output arguments:
    max_score -- maximum score by an opponent
    tie_value -- if you tie, how much is that worth?
    """
    scores = np.zeros((len(idx), points_scored.shape[0]))
    for i_opponent in range(len(idx)):
        scores[i_opponent, :] = np.append(points_scored[0:idx[i_opponent] + 1, idx[i_opponent]],
                                          points_scored[idx[i_opponent], idx[i_opponent] + 1:])
    return scores


def get_week_moneylines(week, season=2017):
    """Collect the historical moneylines for a given week/season

    Keyword argument:
    week -- week to query
    season -- season to query, defaults to 2017

    Output arguments:
    df -- frame with columns (game #, season, week, favorite, home, away, home moneyline, away moneyline, favorite odds)

    """
    # Collect page data
    casino_id = '2&wjb'
    my_url = 'http://m.espn.com/nfl/dailyline?week={}&season={}&seasonType={}'.format(week, season, casino_id)

    # open connection
    u_client = u_req(my_url)
    # download page
    page_html = u_client.read()
    u_client.close()

    # let beautiful soup parse it
    page_soup = Soup(page_html, "html.parser")
    table_data = page_soup.table.find_all("td")

    # Parse Data

    n_games = int(np.floor(len(table_data)/4))
    teams = np.empty(shape=[n_games, 2], dtype='U4')
    mline = np.empty(shape=[n_games, 2])
    favorite_odds = np.empty(shape=[n_games, 1])
    favorite = np.empty(n_games, dtype='U4')
    count = 0
    for i in range(0, len(table_data), 4):
        # a = [table_data[i].contents[idx] for idx in [0, 2]]
        teams[count, :] = np.asarray(team_name2abbrv([table_data[i].contents[idx] for idx in [0, 2]]))
        mline[count, :] = [int(table_data[i+1].contents[idx]) for idx in [0, 2]]
        favorite_odds[count] = probability_favorite_moneyline(mline[count, 0], mline[count, 1])
        if mline[count, 1] < 0:
            favorite[count] = teams[count, 1]
        else:
            favorite[count] = teams[count, 0]
        count = count+1
    # Assign to data frame
    game_colname = []
    for game in range(teams.shape[0]):
        game_colname.append(teams[game, 0]+'_'+teams[game, 1])
    df = pd.DataFrame(columns=['season', 'Week', 'home', 'away', 'favorite', 'homeML', 'awayML', 'favorite_odds'],
                      index=game_colname)
    df['season'] = season
    df['Week'] = week
    df['favorite'] = favorite
    df[['home', 'away']] = teams
    df['favorite_odds'] = favorite_odds
    df[['homeML', 'awayML']] = mline

    return df


def historical_pool_picks(week_number):
    """Return the picks and MNF number from the pool

    Keyword argument:
    week_number -- week of current year (all that is available via cbs)

    Output arguments:
    picks -- dataframe of opponent picks
    mfn -- dataframe of opponent MNF picks

    """
    from selenium import webdriver
    # from selenium.common.exceptions import TimeoutException
    # from selenium.webdriver.support.ui import WebDriverWait
    # from selenium.webdriver.support import expected_conditions as EC
    driver = webdriver.Chrome()

    driver.get('https://www.cbssports.com/login?xurl=http%3A%2F%2Ffunkejim.football.cbssports.com'
               '%2Foffice-pool%2Fmake-picks&master_product=25283')

    username_element = driver.find_element_by_name("userid")
    # omitted from public rep
    password_element = driver.find_element_by_name("password")
    username_element.send_keys("omitted")
    password_element.send_keys("omitted")
    password_element.submit()
    url = 'http://funkejim.football.cbssports.com/office-pool/standings/live/' + str(week_number)

    driver.get(url)
    try:
        team_names = np.array([teams.text.split(' ')[0] for teams in driver.find_elements_by_class_name("bg4")])
    except IndexError:
        pdb.set_trace()
        raise IndexError('Page parsed incorrectly')
    game_colname = []
    for game in range(int(len(team_names)/2)):
        game_colname.append(team_names[2*game]+'_'+team_names[(2*game+1)])
    # The rows NOT including me are found here
    rws = driver.find_elements_by_class_name("bg2")
    # My row
    my_row = driver.find_elements_by_class_name("bgFan")

    # The cells are found with tag_name td
    # The first cell is the name
    name = [iRow.find_element_by_tag_name("td").text for iRow in rws]
    my_name = my_row[0].find_element_by_tag_name("td").text

    # The rest of the cells contain the picks and points
    row_data = [iRow.find_elements_by_tag_name("td") for iRow in rws]
    my_row_data = my_row[0].find_elements_by_tag_name("td")
    table_data = [[cell.text for cell in row[1:]] for row in row_data]
    my_table_data = np.array([[cell.text for cell in my_row_data[1:]]])
    df = pd.DataFrame(table_data, index=name,)
    my_df = pd.DataFrame(my_table_data, index=[my_name])
    mnf = df.iloc[:, -3]
    my_mnf = my_df.iloc[:, -3]
    picks = df.iloc[:, 0:-3]
    my_picks = my_df.iloc[:, :-3]
    mnf.columns = game_colname
    my_mnf.columns = game_colname
    picks.columns = game_colname
    my_picks.columns = game_colname
    return picks, mnf, my_mnf, my_picks


def moneyline_csv(week_number, year=2017, use_get_odds_current_week=False):
    """Write csv files for historical moneylines"""
    fn_moneylines = 'csv/weekMoneylines{}_{}.csv'.format(week_number, year)
    if use_get_odds_current_week:
        df = get_odds_current_week(week_number)
    else:
        df = get_week_moneylines(week_number)
    if df['favorite_odds'].isnull().sum() != 0:
        print('For at least one of these games, no moneyline is set. Fill in from another source!')
    df.to_csv(fn_moneylines, index=True)


def opponent_models_random_forest(week_excluded=None):
    """Build a list of random forest models for all players using build_player_training_frames


    Output arguments:
    clf -- list of trained random forest objects
    players -- the players modeled; indices correspond with indices of models
    variable_names -- variables the model is trained on
    """
    players = get_opponent_names()
    clf = []
    count = 0
    x = None
    for i_player in players:
        x, y, file_not_found_error = build_player_training_frames(i_player, week_excluded=week_excluded)
        if (file_not_found_error[0]) and (i_player == players[0]):
            if file_not_found_error[1] is None:
                print('file {} does not exist'.format(file_not_found_error[2]))
            else:
                print('csv file for picks in week {} not found, frame contains only previous weeks'.format(
                    file_not_found_error[1]))
        clf.append(RandomForestClassifier(n_estimators=50))
        clf[count] = clf[count].fit(x, y)
        count += 1
    variable_names = x.columns
    return clf, players, variable_names


def opponent_models_svm(week_excluded=None):
    """Build a list of random forest models for all players using build_player_training_frames


    Output arguments:
    clf -- list of trained random forest objects
    players -- the players modeled; indices correspond with indices of models
    variable_names -- variables the model is trained on
    """
    players = get_opponent_names()
    clf = []
    count = 0
    x = None
    for i_player in players:
        x, y, file_not_found_error = build_player_training_frames(i_player, week_excluded=week_excluded)
        if (file_not_found_error[0]) and (i_player == players[0]):
            if file_not_found_error[1] is None:
                print('file {} does not exist'.format(file_not_found_error[2]))
            else:
                print('csv file for picks in week {} not found, frame contains only previous weeks'.format(
                    file_not_found_error[1]))
        clf.append(svm.SVC())
        clf[count] = clf[count].fit(x, y)
        count += 1
    variable_names = x.columns
    return clf, players, variable_names


def opponent_models_kneighbors(week_excluded=None):
    """Build a list of random forest models for all players using build_player_training_frames


    Output arguments:
    clf -- list of trained random forest objects
    players -- the players modeled; indices correspond with indices of models
    variable_names -- variables the model is trained on
    """
    players = get_opponent_names()
    clf = []
    count = 0
    x = None
    for i_player in players:
        x, y, file_not_found_error = build_player_training_frames(i_player, week_excluded=week_excluded)
        if (file_not_found_error[0]) and (i_player == players[50]):
            if file_not_found_error[1] is None:
                print('file {} does not exist'.format(file_not_found_error[2]))
            else:
                print('csv file for picks in week {} not found, frame contains only previous weeks'.format(
                    file_not_found_error[1]))
        clf.append(KNeighborsClassifier(100))
        clf[count] = clf[count].fit(x, y)
        count += 1
    variable_names = x.columns
    return clf, players, variable_names


def picks_names2binary(pcks, favorite_team):
    """Convert opponent picks to binary

    Convert the picks of opponents in the pool from the 3 letter team abbreviations to binary; True is favorite picks

    Keyword arguments:
    pcks -- Dataframe of opponent picks
    favorite_team - Pandas Series of favorite team name as output by get_odds_current_week (city name, typically)
    """
    picked_favorite = pcks.isin(list(favorite_team))
    return picked_favorite


def pick_csv(week_number, year=2017):
    """Write csv files for historical opponent picks for games and mnf
    """
    picks, mnf, my_mnf, my_picks = historical_pool_picks(week_number)
    fn_picks = 'csv/picks{}_{}.csv'.format(week_number, year)
    fn_mnf = 'csv/mnf{}_{}.csv'.format(week_number, year)
    fn_my_picks = 'csv/mypicks{}_{}.csv'.format(week_number, year)
    picks.to_csv(fn_picks, index=True)
    mnf.to_csv(fn_mnf, index=True)
    my_picks.to_csv(fn_my_picks, index=True)


def pick_scores(pick_idx, points_scored):
    """Get the scores associated with weekly picks for all possible outcomes

    Keyword arguments:
    opponent_pick_idx -- The outcome index of the picks of opponents
    points_scored -- points scored with each possible prediction for each possible outcome

    Output arguments:
    max_score -- maximum score by an opponent
    tie_value -- if you tie, how much is that worth?
    """

    scores = get_scores_for_prediction(pick_idx, points_scored)
    # Get the maximum score for each outcome
    max_score = np.amax(scores, 0).astype(np.int8)
    # Get the number of players with max score for each outcome
    n_max = np.sum(np.equal(np.tile(max_score, (scores.shape[0], 1)), scores), 0, dtype=np.int8)
    tie_value = (1./(n_max+1)).astype(np.float16)
    return max_score, tie_value


def predicted_pick_values(clf, df_moneylines):
    """
    Use a list of classifiers to determine the top 50 sets of picks and their values
    :param clf: List of classifiers
    :param df_moneylines: df_moneylines as read in from the csv file
    """
    x = engineered_features(df_moneylines)
    picks_pred = generate_picks(clf, x)
    n_opponents = picks_pred.shape[0]
    n_games = picks_pred.shape[1]
    all_outcomes, points_scored = combos(n_games)
    outcome_odds = all_outcome_odds(df_moneylines['favorite_odds'], all_outcomes)
    total_value_pick_pred, max_score_pred, _, _ = calculate_pick_values(
        all_outcomes, points_scored, outcome_odds, picks_pred, df_moneylines)
    ind = best_pick_indices(total_value_pick_pred)
    results = []
    for i_pick in range(len(ind)):
        results.append(get_pick_odds(
            ind[0], points_scored, max_score_pred, outcome_odds, total_value_pick_pred, n_opponents, all_outcomes))
    return total_value_pick_pred, ind, picks_pred, all_outcomes, results


def print_opponent_pick_odds(opponent_pick_idx, points_scored, max_score, outcome_odds, total_value_pick, n_opponents,
                             picks_made, my_pick_idx, all_outcomes):
    individual_odds = np.array([])
    for i_opponent in range(n_opponents):
        results = get_pick_odds(opponent_pick_idx[i_opponent], points_scored, max_score, outcome_odds, total_value_pick,
                                n_opponents, all_outcomes)
        individual_odds = np.append(individual_odds, results["total_value_this_pick"])
        print('{:<13}: {:.2f}'.format(picks_made.index[i_opponent], results["normalized"]))

    results = get_pick_odds(my_pick_idx, points_scored, max_score, outcome_odds, total_value_pick,
                            n_opponents, all_outcomes)
    individual_odds = np.append(individual_odds, results["total_value_this_pick"])
    print(np.sum(individual_odds))
    print(individual_odds)


def print_pick_outcome_odds(results):
    """Print the odds of winning, tie and total value for a given pick, given the existing picks and odds"""
    print(results["n_winning_outcomes"], "winning outcomes and", results["n_tying_outcomes"], "tying outcomes")
    print('The odds of winning outright are: {:.2f}%'.format(100*results["winning_odds"]))
    print('The odds of tying are: {:.2f}%'.format(100*results["tying_odds"]))
    print('{} underdog picks'.format(results['n_underdog_picks']))
    if results["roi"] > 0:
        print('{:0.3f}: Total value\n{:2.1f}% greater than expected return'.format(
            results["total_value_this_pick"], results["roi"]))
    else:
        print('{:0.3f}: Total value\n{:2.1f}% worse than base return'.format(
            results["total_value_this_pick"], -results["roi"]))


def probability_favorite_moneyline(value1, value2):
    """Get the probability of the favorite winning based on moneylines"""

    def get_prob(val):
        """Calculate implied probability for a given moneyline"""
        # If underdog
        if val > 0:
            prob = 100./(val+100.)
        # If favorite
        elif val < 0:
            val = val*-1
            prob = val/(val+100.)
        else:
            raise ValueError
        return prob
    p1_vigged = get_prob(value1)
    p2_vigged = get_prob(value2)
    vig = p1_vigged+p2_vigged
    p1 = p1_vigged/vig
    p2 = p2_vigged/vig
    p_fav = max(p1, p2)
    return p_fav


def probability_favorite_spread(spr):
    """Get the probability based on empirical results for spreads"""

    idx = int(spr*2.)
    idx = min(idx, 34)
    p = [50., 50.5, 51.3, 52.5, 53.5, 54.5, 59.4, 64.3, 65.8, 67.3, 68.1, 69.0, 70.7, 72.4, 75.2, 78.1, 79.1, 80.2,
         80.7, 81.1, 83.6, 86.0, 87.1, 88.2, 88.5, 88.7, 89.3, 90.0, 92.4, 94.9, 95.6, 96.3, 98.1, 99.8, 100]
    try:
        odds = 0.01*p[idx]
    except IndexError:
        odds = 100
    return odds


def record2float(wins,losses):
    games_played= wins+losses
    regressed_percentage = (wins+6)/(12+games_played)
    expected_additional_wins = (16-games_played)*regressed_percentage
    expected_win_percentage = (wins+expected_additional_wins)/16
    return expected_win_percentage



def save_top_pick_visualizations(start_week, end_week, year=2017):
    for week_number in range(start_week, end_week):

        picks = pd.read_csv('csv/picks{}_{}.csv'.format(week_number, year), index_col=0)
        my_picks = pd.read_csv('csv/mypicks{}_{}.csv'.format(week_number, year), index_col=0)
        picks_made = picks.loc[np.sum(picks.isnull(), axis=1) < 5, :]  # removes people who didn't get picks in

        df_moneylines = pd.read_csv('csv/weekMoneylines{}_{}.csv'.format(week_number, year), index_col=0)
        df_moneylines = validate_and_align_input_data(df_moneylines, picks_made)

        all_outcomes, points_scored = combos(df_moneylines['favorite_odds'])
        outcome_odds = all_outcome_odds(df_moneylines['favorite_odds'], all_outcomes)
        total_value_pick, max_score, opponent_pick_idx, my_pick_idx = calculate_pick_values(
            all_outcomes, points_scored, outcome_odds, picks_made, my_picks, df_moneylines)

        # Get indices of the top 5 picks
        ind = np.argpartition(total_value_pick, -5)[-5:]
        ind = ind[np.argsort(-total_value_pick[ind])]
        # top_5_values = total_value_pick[ind]

        # underdog_picks = ~all_outcomes[:, ind]

        # Store results
        risks_to_take = []
        team_to_pick_against = []
        for i_pick in range(5):
            risks_to_take.append(df_moneylines['favorite_odds'][~all_outcomes[:, ind[i_pick]]])
            team_to_pick_against.append(df_moneylines['favorite'][~all_outcomes[:, ind[i_pick]]])

        # Plot results
        for i_pick in range(5):
            ax = plt.subplot(5, 1, i_pick + 1)
            plt.plot(df_moneylines['favorite_odds'], np.zeros(df_moneylines['favorite_odds'].shape), 'ok')
            ax.set_xlim([.5, .95])
            if i_pick != 4:
                ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_ticks([])
            ax.set_ylabel('{}'.format(len(risks_to_take[i_pick])))
            plt.plot(risks_to_take[i_pick], np.zeros(risks_to_take[i_pick].shape), 'or')

        plt.subplot(5, 1, 1)
        plt.title('week {}'.format(week_number))
        plt.savefig('/Users/ahakso/Downloads/week{}.png'.format(week_number), bbox_inches='tight')
        plt.close()


def team_name2abbrv(names):
    """ Convert from a variety of representations of a team name to the three letter abbreviation

    Each element of the input object is converted to abbreviation.
    keyword arguments:
    names -- team name(s). Can be either a single team name or a list of names
    """
    abbrv = []
    if type(names) != list:
        names = [names]
    teams = [
        ['ARI', 'Arizona', 'Cardinals', 'Arizona Cardinals'],
        ['ATL', 'Atlanta', 'Falcons', 'Atlanta Falcons'],
        ['BAL', 'Baltimore', 'Ravens', 'Baltimore Ravens'],
        ['BUF', 'Buffalo', 'Bills', 'Buffalo Bills'],
        ['CAR', 'Carolina', 'Panthers', 'Carolina Panthers'],
        ['CHI', 'Chicago', 'Bears', 'Chicago Bears'],
        ['CIN', 'Cincinnati', 'Bengals', 'Cincinnati Bengals'],
        ['CLE', 'Cleveland', 'Browns', 'Cleveland Browns'],
        ['DAL', 'Dallas', 'Cowboys', 'Dallas Cowboys'],
        ['DEN', 'Denver', 'Broncos', 'Denver Broncos'],
        ['DET', 'Detroit', 'Lions', 'Detroit Lions'],
        ['GB', 'Green Bay', 'Packers', 'Green Bay Packers', 'G.B.', 'GNB'],
        ['HOU', 'Houston', 'Texans', 'Houston Texans'],
        ['IND', 'Indianapolis', 'Colts', 'Indianapolis Colts'],
        ['JAC', 'Jacksonville', 'Jaguars', 'Jacksonville Jaguars', 'JAX'],
        ['KC', 'Kansas City', 'Chiefs', 'Kansas City Chiefs', 'K.C.', 'KAN'],
        ['LAC', 'Chargers', 'Los Angeles Chargers', 'LA Chargers'],
        ['LAR', 'Rams', 'Los Angeles Rams', 'L.A.', 'LA Rams'],
        ['MIA', 'Miami', 'Dolphins', 'Miami Dolphins'],
        ['MIN', 'Minnesota', 'Vikings', 'Minnesota Vikings'],
        ['NE', 'New England', 'Patriots', 'New England Patriots', 'N.E.', 'NWE'],
        ['NO', 'New Orleans', 'Saints', 'New Orleans Saints', 'N.O.', 'NOR'],
        ['NYG', 'NY Giants', 'Giants', 'New York Giants', 'N.Y.G.'],
        ['NYJ', 'NY Jets', 'Jets', 'New York Jets', 'N.Y.J.'],
        ['OAK', 'Oakland', 'Raiders', 'Oakland Raiders'],
        ['PHI', 'Philadelphia', 'Eagles', 'Philadelphia Eagles'],
        ['PIT', 'Pittsburgh', 'Steelers', 'Pittsburgh Steelers'],
        ['SEA', 'Seattle', 'Seahawks', 'Seattle Seahawks'],
        ['SF', 'San Francisco', '49ers', 'San Francisco 49ers', 'S.F.', 'SFO'],
        ['STL', 'St. Louis', 'Rams', 'St. Louis Rams', 'S.T.L.'],
        ['TB', 'Tampa Bay', 'Buccaneers', 'Tampa Bay Buccaneers', 'T.B.', 'TAM'],
        ['TEN', 'Tennessee', 'Titans', 'Tennessee Titans'],
        ['WAS', 'Washington', 'Redskins', 'Washington Redskins', 'WSH'],
    ]
    teams = [[y.upper() for y in x] for x in teams]
    names = [x.upper() for x in names]
    for name in names:
        abbrv.append(teams[np.nonzero([name in x for x in teams])[0][0]][0])
    return abbrv


def validate_and_align_input_data(ml, picks):
    """Align the games in the moneylines frame with the order as displayed in the picks frame
The moneylines frame may be arbitrarily ordered from the two different sources, while the cbssports site
    provides the reference order.
    """
    pd.options.mode.chained_assignment = None
    ref_order = picks.columns
    to_shuffle = ml.index
    n_games = len(ref_order)
    if ml.index.isin(picks.columns).sum() != len(ml.index):
        ml_missing_from_picks = np.nonzero(np.logical_not(ml.index.isin(picks.columns)))[0]
        picks_missing_from_ml = np.nonzero(np.logical_not(picks.columns.isin(ml.index)))[0]
        if len(ml_missing_from_picks) > 0:
            raise MissingGame('Picks file is missing game(s): {}\nAdjust in csv'.format(
                ' '.join([x for x in ml.index[ml_missing_from_picks]])))
        elif len(picks_missing_from_ml) > 0:
            raise MissingGame('moneylines file is missing game(s) {}\nAdjust in csv'.format(
                ' '.join([x for x in picks.columns[picks_missing_from_ml]])))
    shuffle_index = np.zeros(n_games, dtype=np.int)
    for i_row in range(n_games):
        try:
            shuffle_index[i_row] = np.where(ref_order[i_row] == to_shuffle)[0][0]
        except IndexError:
            print('The {} at {} game in week {} does not have a moneyline provided by the casino'.format(
                ref_order[i_row].split('_')[0], ref_order[i_row].split('_')[1], ml['Week'][0]))
    ml = ml.iloc[shuffle_index, :]
    return ml


def variable_importance(clf, variable_names):
    values = []
    players = get_opponent_names()
    for i_player in range(len(clf)):
        values.append(clf[i_player].feature_importances_)
    values = pd.DataFrame(values, index=players, columns=variable_names)
    return values


def write_hdf5_combos():
    f = h5py.File('cache/points_scored.h5', 'r')
    for n_games in range(9, 17):
        _, points_scored = combos(n_games)
        f.create_dataset(str(n_games), data=points_scored)
    f.flush()
    f.close()
