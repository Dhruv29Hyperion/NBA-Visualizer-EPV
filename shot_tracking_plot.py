"""
Scripts for extracting the shot probability of a shot
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# To get the offensive rating of the shooter and the defensive rating of the closest defender
ratings = pd.read_csv('C:/Users/Dhruv/PycharmProjects/basketballPlayOptimisation/Tracking_Data/player'
                      '/player_ratings.csv')

fg_percentage_zones = pd.read_csv(
    'C:/Users/Dhruv/PycharmProjects/basketballPlayOptimisation/Tracking_Data/player/player_shot_percentages.csv')


def load_model_and_predict(feature_vector):
    # Load the trained model
    model = tf.keras.models.load_model('shot_model_v2.keras')

    # Load the scaler used for normalization and set its mean and scale directly
    scaler = StandardScaler()
    scaler.mean_ = np.load('shot_scaler_mean.npy')
    scaler.scale_ = np.load('shot_scaler_scale.npy')

    # Ensure feature_vector is numpy array and 2D
    feature_vector = np.array(feature_vector, dtype=np.float32)

    # Select columns to scale
    columns_to_scale = [0, 1, 4, 5, 7, 8, 9, 10]

    # Scale the selected columns
    feature_vector[columns_to_scale] = scaler.transform(feature_vector[columns_to_scale].reshape(1, -1)).reshape(-1)

    feature_vector = feature_vector.reshape(1, -1)

    # Make predictions
    prediction = model.predict(feature_vector)

    return prediction[0][0]


def predict_output(feature_vector):
    # Load the encoders
    encoder_shot_zone = pickle.load(open('encoder_shot_zone.pkl', 'rb'))
    encoder_shot_area = pickle.load(open('encoder_shot_area.pkl', 'rb'))
    encoder_shot_range = pickle.load(open('encoder_shot_range.pkl', 'rb'))

    # Load the scaler
    scaler = StandardScaler()
    scaler.mean_ = np.load('zone_scaler_mean.npy')
    scaler.scale_ = np.load('zone_scaler_scale.npy')

    # Coverting the feature vector to a numpy array
    feature_vector = np.array(feature_vector)

    # Convert features to float32
    feature_vector = feature_vector.astype(np.float32)

    # Normalize the last column of the feature vector
    feature_vector[-1] = scaler.transform(feature_vector[-1].reshape(1, -1)).reshape(-1)

    feature_vector = feature_vector.reshape(1, -1)

    # Predict using the model
    model = tf.keras.models.load_model('zone_model_v2.keras')

    predictions = model.predict(feature_vector)

    # Decode the predicted values
    predicted_shot_zone = encoder_shot_zone.inverse_transform([np.argmax(predictions[0])])[0]
    predicted_shot_area = encoder_shot_area.inverse_transform([np.argmax(predictions[1])])[0]
    predicted_shot_range = encoder_shot_range.inverse_transform([np.argmax(predictions[2])])[0]

    return [predicted_shot_zone, predicted_shot_area, predicted_shot_range]


def extract_shot_probability(game, details, previous_details):
    """
    Extract the shot probability of a shot

    Args:
    game: Game object containing the tracking data and the play by play data of the game
    details: List containing the details of the moment in the game
    previous_details: List containing the details of the previous moment in the game

    Returns:
    shot_probability: The probability of the shot going in for all the offensive players ->[[player1, probability1],...]
    """

    shot_probability = {}

    # Check if the details are valid
    if len(details[1]) != len(previous_details[1]) or len(details[1]) != 11:
        return shot_probability

    # Get the id of the shooter
    ball = [details[1][0], details[2][0]]
    possible_ball_handler_coords = [[details[1][i], details[2][i]] for i in range(1, len(details[1]))]
    ball_handler_coords = min(possible_ball_handler_coords, key=lambda x: np.linalg.norm(np.array(ball) - np.array(x)))
    ball_handler_index = details[1].index(ball_handler_coords[0])
    ball_handler = details[10][ball_handler_index]

    # Shot clock time
    shot_clock = int(details[6])

    # Get the quarter and minute of the game
    quarter = details[5]
    minutes = int(details[7].split(':')[0])
    seconds = int(details[7].split(':')[1])

    home_team_ids = [i for i in details[10] if i in list(game.player_ids.values())[:13]]
    away_team_ids = [i for i in details[10] if i in list(game.player_ids.values())[13:]]

    # Get the offensive team and determine the basket coordinates [5.35, -25] or [88.65, -25]
    if ball_handler in home_team_ids:
        offensive_team = home_team_ids
        defensive_team = away_team_ids

    else:
        offensive_team = away_team_ids
        defensive_team = home_team_ids

    offense = [[details[1][details[10].index(teammate)],
                details[2][details[10].index(teammate)]]
               for teammate in offensive_team]

    defense = [[details[1][details[10].index(teammate)],
                details[2][details[10].index(teammate)]]
               for teammate in defensive_team]

    # Below is the logic to determine the basket coordinates based on the quarter and the team shooting

    x_pos = np.array(details[1])
    if len(x_pos) != 11:
        return shot_probability

    # This is so that the basket is always on the right side of the court. Doesn't work all the time, but works for
    # most cases. Usually works when the defenses and offenses are set up correctly on 1 half of the court.
    # So doesn't always capture transition plays, and that's okay as
    # the degree of difficulty for making a decision is less.

    basket = None

    if game.flip_direction:
        if (x_pos < 47).all() and quarter in [1, 2] and offensive_team == away_team_ids:  # away
            basket = [5.35, -25]
        if (x_pos > 47).all() and quarter in [3, 4] and offensive_team == away_team_ids:  # away
            basket = [88.65, -25]
        if (x_pos < 47).all() and quarter in [3, 4] and offensive_team == home_team_ids:  # home
            basket = [5.35, -25]
        if (x_pos > 47).all() and quarter in [1, 2] and offensive_team == home_team_ids:
            basket = [88.65, -25]

    if (x_pos > 47).all() and quarter in [1, 2] and offensive_team == away_team_ids:  # away
        basket = [5.35, -25]
    if (x_pos < 47).all() and quarter in [3, 4] and offensive_team == away_team_ids:  # away
        basket = [88.65, -25]
    if (x_pos > 47).all() and quarter in [3, 4] and offensive_team == home_team_ids:  # home
        basket = [5.35, -25]
    if (x_pos < 47).all() and quarter in [1, 2] and offensive_team == home_team_ids:  #home
        basket = [88.65, -25]

    if basket is None:
        return shot_probability

    court_center = [47, -25]  # Assuming this is the center of the court along the x-axis

    # Get the score margin of the game
    score_margin = None
    game_time = details[0]

    while score_margin is None:
        for game_second in range(game_time - 1, game_time + 1):
            for index, row in game.pbp[game.pbp.game_time ==
                                       game_second].iterrows():
                # check if it isn't empty
                if pd.isna(row['SCOREMARGIN']) is False:
                    score_margin = row['SCOREMARGIN']
                    break
        game_time += 1

    if score_margin == 'TIE' or pd.isna(score_margin):
        score_margin = 0
    elif offensive_team == away_team_ids:
        score_margin = -int(score_margin)
    else:
        score_margin = int(score_margin)

    # how many points was the shot attempted for
    three_pt_zones = ['Above the Break 3', 'Backcourt', 'Left Corner 3', 'Right Corner 3']
    two_pt_zones = ['In The Paint (Non-RA)', 'Mid-Range', 'Restricted Area']

    for player in offensive_team:

        player_index = details[10].index(player)
        player_coords = [details[1][player_index], details[2][player_index]]

        player_name = list({i for i in game.player_ids if game.player_ids[i] == player})[0]

        player_offensive_rating = ratings['OFFENSIVE_RATING'].mean()

        if player_name in ratings['PLAYER_NAME'].values:
            player_offensive_rating = ratings[ratings['PLAYER_NAME'] == player_name]['OFFENSIVE_RATING'].values[0]

        dist = np.linalg.norm(np.array(player_coords) - np.array(basket))

        closest_defender_coord = defense[
            np.argmin([np.linalg.norm(np.array(player_coords) - np.array(defender))
                       for defender in defense])]

        closest_defender_dist = np.linalg.norm(np.array(player_coords) - np.array(closest_defender_coord))

        closest_defender_index = details[1].index(closest_defender_coord[0])

        closest_defender = details[10][closest_defender_index]

        closest_defender_name = list({i for i in game.player_ids if game.player_ids[i] == closest_defender})[0]

        closest_defender_defensive_rating = ratings['DEFENSIVE_RATING'].mean()

        if closest_defender_name in ratings['PLAYER_NAME'].values:
            closest_defender_defensive_rating = \
                ratings[ratings['PLAYER_NAME'] == closest_defender_name]['DEFENSIVE_RATING'].values[0]

        player_basket_vector = np.array(basket) - np.array(player_coords)
        center_basket_vector = np.array(basket) - np.array(court_center)
        shot_angle = np.arccos(np.dot(player_basket_vector, center_basket_vector) /
                               (np.linalg.norm(player_basket_vector) * np.linalg.norm(center_basket_vector))
                               )

        player_defender_vector = np.array(closest_defender_coord) - np.array(player_coords)
        shot_defender_angle = np.arccos(np.dot(player_basket_vector, player_defender_vector) /
                                        (np.linalg.norm(player_basket_vector) * np.linalg.norm(player_defender_vector))
                                        )

        num_close_defenders = sum([1 for defender in defense
                                   if np.linalg.norm(np.array(player_coords) - np.array(defender)) < 10])

        delta_x = np.array(details[1]) - np.array(previous_details[1])
        delta_y = np.array(details[2]) - np.array(previous_details[2])
        delta_time = details[9] - previous_details[9]
        player_velocity = np.linalg.norm([delta_x[player_index], delta_y[player_index]]) / delta_time

        # Get the zone classification
        zone = predict_output([player_coords[0], player_coords[1], dist])

        # Default shot attempted is set from 2 points zone
        points_attempted = 2
        if zone[0] in three_pt_zones:
            points_attempted = 3
        elif zone[0] in two_pt_zones:
            points_attempted = 2

        fg_percentage = 34.6  # Average field goal percentage in the NBA

        if player_name in fg_percentage_zones['PLAYER_NAME'].values:
            fg_percentage = fg_percentage_zones[(fg_percentage_zones['SHOT_ZONE_BASIC'] == zone[0]) &
                                                (fg_percentage_zones['SHOT_ZONE_AREA'] == zone[1]) &
                                                (fg_percentage_zones['SHOT_ZONE_RANGE'] == zone[2])]['SHOT_PERCENTAGE']

        if fg_percentage.empty or fg_percentage.values[0] == 0:
            fg_percentage = 34.6
        else:
            fg_percentage = fg_percentage.values[0]

        defender_velocity = np.linalg.norm(
            [delta_x[closest_defender_index], delta_y[closest_defender_index]]) / delta_time

        feature_vector = ([player_offensive_rating, dist, player_coords[0], player_coords[1],
                           shot_angle, player_velocity, fg_percentage,
                           closest_defender_defensive_rating,
                           closest_defender_dist, shot_defender_angle, defender_velocity, num_close_defenders,
                           shot_clock, score_margin, quarter, minutes, seconds, points_attempted])

        # Get the shot difficulty
        shot_difficulty = load_model_and_predict(feature_vector)

        points_expected = shot_difficulty * points_attempted

        shot_probability[player] = points_expected

    return shot_probability
