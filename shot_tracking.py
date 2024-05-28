"""
Scripts for extracting shot_features from the NBA tracking data
"""

import os
import pickle
import numpy as np
import pandas as pd
from simulate import Game

# To get the offensive rating of the shooter and the defensive rating of the closest defender
ratings = pd.read_csv('C:/Users/Dhruv/PycharmProjects/basketballPlayOptimisation/Tracking_Data/player'
                      '/player_ratings.csv')

# To get the field goal % of the shooter from the specific zone he is shooting from
zones = pd.read_csv('C:/Users/Dhruv/PycharmProjects/basketballPlayOptimisation/Tracking_Data/shots/shots.csv')

fg_percentage = pd.read_csv(
    'C:/Users/Dhruv/PycharmProjects/basketballPlayOptimisation/Tracking_Data/player/player_shot_percentages.csv')


def extract_games():
    """
    Extract games from allgames.txt

    Returns:
        list: list of games.  Each element is list is
            [date, home_team, away_team]
            an example element: ['01.01.2016', 'TOR', 'CHI']
    """

    games = []
    with open('allgames.txt', 'r') as game_file:
        for line in game_file:
            game = line.strip().split('.')
            date = "{game[0]}.{game[1]}.{game[2]}".format(game=game)
            away = game[3]
            home = game[5]
            games.append([date, home, away])
    return games


def get_features(date, home_team, away_team, write_file=True,
                 write_score=False, write_game=False):
    """
    Calculates shot_features1 for each frame in game

    Args:
        date (str): date of game in form 'MM.DD.YYYY'.  Example: '01.01.2016'
        home_team (str): home team in form 'XXX'. Example: 'TOR'
        away_team (str): away team in form 'XXX'. Example: 'CHI'
        write_file (bool): If True, write pickle file of spacing
            statistics into data/spacing directory
        write_score (bool): If True, write pickle file of game score
            into data/score directory
        write_game (bool): If True, write pickle file of tracking data
            into data/game directory
            Note: This file is ~100MB.

        Returns:
        tuple: tuple of data:
            (
            event : event number of the shot -> int,
            shooter : id of the shooter -> int,
            shooter_name : name of the shooter -> str,
            shooter_offensive_rating : offensive rating of the shooter -> float,
            dist : distance of the shot which is basically shooter to basket distance -> float,
            x : x coordinate of the shot -> float,
            y : y coordinate of the shot -> float,
            shot_angle : angle of the shot w.r.t court center -> float,
            shooter_velocity : velocity of the shooter in ft/msec -> float,
            fg_percentage_zone : field goal percentage of the shooter from the specific zone he is shooting from -> float,
            closest_defender : id of the closest defender -> int,
            closest_defender_name : name of the closest defender -> str,
            closest_defender_defensive_rating : defensive rating of the closest defender -> float,
            closest_defender_dist : distance of the closest defender from the shooter -> float,
            closest_defender_angle : angle of the closest defender w.r.t shot trajectory -> float,
            closest_defender_velocity : velocity of the closest defender in ft/msec -> float,
            num_close_defenders : number of defenders within 4 feet of the shooter -> int,
            shot_clock : time left on the shot clock -> float,
            score_margin : score margin of the game -> int,
            quarter : quarter of the game -> int,
            minutes : minutes remaining in the quarter -> int,
            seconds : seconds remaining in the quarter -> int,
            make : 1 if the shot was made, 0 if missed -> int
            )
    """
    game = Game(date, home_team, away_team)

    # Extract shot_features for each frame
    features = [['event', 'shooter', 'shooter_name', 'shooter_offensive_rating', 'dist', 'x', 'y', 'shot_angle',
                 'shooter_velocity', 'fg_percentage_zone', 'closest_defender', 'closest_defender_name',
                 'closest_defender_defensive_rating', 'closest_defender_dist', 'closest_defender_angle',
                 'closest_defender_velocity', 'num_close_defenders', 'shot_clock', 'score_margin', 'quarter', 'make']]

    zone = [['x', 'y', 'dist', 'SHOT_ZONE_BASIC' , 'SHOT_ZONE_AREA', 'SHOT_ZONE_RANGE']]

    filename = ("{date}-{away_team}-"
                "{home_team}.pkl").format(date=date,
                                          away_team=away_team,
                                          home_team=home_team)

    # Do not recalculate spacing data if already saved to disk
    if filename in os.listdir('./sfeatures'):
        return

    # Write the features to a csv file
    # if write_file:
    #     with open("shot_features/{filename}".format(filename=filename), 'w') as myfile:
    #         myfile.write("event, shooter, shooter_name, shooter_offensive_rating, dist, x, y, shot_angle, "
    #                      "shooter_velocity, fg_percentage_zone, closest_defender, closest_defender_name, "
    #                      "closest_defender_defensive_rating, closest_defender_dist, closest_defender_angle, "
    #                      "closest_defender_velocity, num_close_defenders, shot_clock, score_margin, quarter, make\n")

    # Get all shot frames where EVENTMSGTYPE == 1 (shot-made) or 2 (shot missed)
    shot_frames = sorted(game.pbp[game.pbp['EVENTMSGTYPE'] == 1]['EVENTNUM'].tolist() +
                         game.pbp[game.pbp['EVENTMSGTYPE'] == 2]['EVENTNUM'].tolist())

    for frame in shot_frames:

        # Get the frame of the shot
        shot_frame = game.get_play_frames(event_num=frame)[1]

        # Get the moment details as per the frame
        details = game._get_moment_details(shot_frame)

        temp = shot_frame
        # format pctimestring to be in the format of "%02d:%02d" of the play by play
        pbp_time = game.pbp[game.pbp['EVENTNUM'] == frame]['PCTIMESTRING'].values[0]
        pbp_time = pbp_time.split(':')
        pbp_time = "{:02d}:{:02d}".format(int(pbp_time[0]), int(pbp_time[1]))
        # If the game clock doesn't in the play by play doesn't match the game clock in the tracking data
        # (margin of + or -1 second), then continue
        counter = 0
        while pbp_time != details[7]:
            counter += 1
            temp -= 1
            details = game._get_moment_details(temp)
            if temp == 0 or counter > 5:
                break

        if temp == 0 or counter > 5:
            continue

        # Get the id of the shooter
        shooter = game.pbp[game.pbp['EVENTNUM'] == frame]['PLAYER1_ID'].values[0]
        shooter_name = game.pbp[game.pbp['EVENTNUM'] == frame]['PLAYER1_NAME'].values[0]

        filtered_zones = zones[(zones['PLAYER_NAME'] == shooter_name) & (zones['GAME_ID'] == int(game.game_id)) & (
                zones['GAME_EVENT_ID'] == frame)]

        while shooter not in details[10]:
            temp -= 1
            details = game._get_moment_details(temp)
            if temp == 0:
                break
        shooter_index = details[10].index(shooter)

        # Shot clock time
        shot_clock = int(details[6])
        # Get the quarter and minute of the game
        if not filtered_zones['PERIOD'].empty:
            quarter = filtered_zones['PERIOD'].values[0]
            minutes = filtered_zones['MINUTES_REMAINING'].values[0]
            seconds = filtered_zones['SECONDS_REMAINING'].values[0]
        else:
            quarter = details[5]
            minutes = int(pbp_time.split(':')[0])
            seconds = int(pbp_time.split(':')[1])

        # Get the x and y coordinates of the shooter
        shooter_coords = [details[1][shooter_index], details[2][shooter_index]]

        # how many points was the shot attempted for
        three_pt_zones = ['Above the Break 3', 'Backcourt', 'Left Corner 3', 'Right Corner 3']
        two_pt_zones = ['In The Paint (Non-RA)', 'Mid-Range', 'Restricted Area']

        # Default shot attempted is set from 2 points zone
        points_attempted = 2
        if not filtered_zones['SHOT_ZONE_BASIC'].empty:
            if filtered_zones['SHOT_ZONE_BASIC'].values[0] in three_pt_zones:
                points_attempted = 3
            elif filtered_zones['SHOT_ZONE_BASIC'].values[0] in two_pt_zones:
                points_attempted = 2

        # Determine the basket coordinates based on which the basket is closer to the shooter [5.35, 25] or [88.65, 25]
        basket = [5.35, -25] if shooter_coords[0] < 47 else [88.65, -25]

        # Get the distance of the shot
        # if not filtered_zones['SHOT_DISTANCE'].empty:
        #     dist = filtered_zones['SHOT_DISTANCE'].values[0]
        # else:
        #     dist = np.linalg.norm(np.array(shooter_coords) - np.array(basket))

        dist = np.linalg.norm(np.array(shooter_coords) - np.array(basket))

        home_team_ids = [i for i in details[10] if i in list(game.player_ids.values())[:13]]
        away_team_ids = [i for i in details[10] if i in list(game.player_ids.values())[13:]]

        # Get the offensive team
        if shooter in home_team_ids:
            offensive_team = home_team_ids
            defensive_team = away_team_ids
        else:
            offensive_team = away_team_ids
            defensive_team = home_team_ids

        # Get the x and y coordinates of the defenders
        defender_coords = [[details[1][details[10].index(defender)],
                            details[2][details[10].index(defender)]]
                           for defender in defensive_team]

        # Get the distance of the closest defender
        closest_defender_coord = defender_coords[
            np.argmin([np.linalg.norm(np.array(shooter_coords) - np.array(defender))
                       for defender in defender_coords])]
        closest_defender_dist = np.linalg.norm(np.array(shooter_coords) - np.array(closest_defender_coord))

        closest_defender_index = details[1].index(closest_defender_coord[0])
        # Get the player id of the closest defender
        closest_defender = details[10][closest_defender_index]

        # Extract the name of the closest defender from the player_ids dictionary and get the value
        closest_defender_name = list({i for i in game.player_ids if game.player_ids[i] == closest_defender})[0]

        # Calculate the angle between the shooter, the basket, and the center of the court
        court_center = [47, -25]  # Assuming this is the center of the court along the x-axis
        shooter_basket_vector = np.array(basket) - np.array(shooter_coords)
        center_basket_vector = np.array(basket) - np.array(court_center)
        shot_angle = np.arccos(np.dot(shooter_basket_vector, center_basket_vector) /
                               (np.linalg.norm(shooter_basket_vector) * np.linalg.norm(center_basket_vector)))

        # Calculate the angle between the shooter and the closest defender and the basket
        shooter_defender_vector = np.array(closest_defender_coord) - np.array(shooter_coords)
        shot_defender_angle = np.arccos(np.dot(shooter_basket_vector, shooter_defender_vector) /
                                        (np.linalg.norm(shooter_basket_vector) * np.linalg.norm(
                                            shooter_defender_vector)))

        # Get the number of defenders within 4 feet of the shooter
        num_close_defenders = sum([1 for defender in defender_coords
                                   if np.linalg.norm(np.array(shooter_coords) - np.array(defender)) < 10])

        # Match the name of the shooter, if the shooter name isn't there, give the average of all ratings
        # Check if the shooter's name exists in the DataFrame
        if shooter_name in ratings['PLAYER_NAME'].values:
            # If it exists, get the shooter's offensive rating
            shooter_offensive_rating = ratings[ratings['PLAYER_NAME'] == shooter_name]['OFFENSIVE_RATING'].iloc[0]
        else:
            # If it doesn't exist, set the shooter's offensive rating to the average of all ratings
            shooter_offensive_rating = ratings['OFFENSIVE_RATING'].mean()

        # Do the same for the defender
        if closest_defender_name in ratings['PLAYER_NAME'].values:
            defender_defensive_rating = \
                ratings[ratings['PLAYER_NAME'] == closest_defender_name]['DEFENSIVE_RATING'].iloc[0]
        else:
            defender_defensive_rating = ratings['DEFENSIVE_RATING'].mean()

        if not filtered_zones['SHOT_ZONE_BASIC'].empty:
            fg_percentage_zone = fg_percentage[(fg_percentage['PLAYER_NAME'] == shooter_name) &
                                               (fg_percentage['SHOT_ZONE_BASIC'] ==
                                                filtered_zones['SHOT_ZONE_BASIC'].values[0]) &
                                               (fg_percentage['SHOT_ZONE_AREA'] ==
                                                filtered_zones['SHOT_ZONE_AREA'].values[0]) &
                                               (fg_percentage['SHOT_ZONE_RANGE'] ==
                                                filtered_zones['SHOT_ZONE_RANGE'].values[0])]['SHOT_PERCENTAGE'].iloc[0]
        else:
            # According to Stanford University, 95% of the NBA have a true talent FG% within 0.346 and 0.556
            # If the player isn't listed, we are assuming he isn't a regular and thus falls in the bottom talent pool
            fg_percentage_zone = 34.6

        # Get the score margin of the game
        score_margin = game.pbp[game.pbp['EVENTNUM'] == frame]['SCOREMARGIN'].values[0]

        # if the score margin is float nan and not found, go back to the previous frames until a score margin is found

        temp_event = frame
        if pd.isna(score_margin):
            while pd.isna(score_margin) and temp_event > 0:
                temp_event -= 1
                if temp in game.pbp['EVENTNUM'].values:
                    score_margin = game.pbp[game.pbp['EVENTNUM'] == temp_event]['SCOREMARGIN'].values[0]
                if temp_event < frame - 10:
                    # Take out from previous features
                    score_margin = features[-1][-3]

        if score_margin == 'TIE' or pd.isna(score_margin):
            score_margin = 0
        elif offensive_team == away_team_ids:
            score_margin = -int(score_margin)
        else:
            score_margin = int(score_margin)

        # Get the make -> 1 if the shot was made, 0 if missed
        make = 1 if game.pbp[game.pbp['EVENTNUM'] == frame]['EVENTMSGTYPE'].values[0] == 1 else 0

        # Get the velocity of the shooter
        if temp > 0:
            previous_details = game._get_moment_details(temp - 1)
        else:
            previous_details = game._get_moment_details(temp)

        # For inconsistent number of players listed in the tracking data, go back to the previous frame under the
        # assumption that velocity barely changes in 1 frame
        temp = temp - 1
        while len(details[1]) != len(previous_details[1]) or len(details[1]) != 11:
            details = previous_details
            temp -= 1
            previous_details = game._get_moment_details(temp)

        delta_x = np.array(details[1]) - np.array(previous_details[1])
        delta_y = np.array(details[2]) - np.array(previous_details[2])
        delta_time = details[9] - previous_details[9]
        shooter_velocity = np.linalg.norm([delta_x[shooter_index], delta_y[shooter_index]]) / delta_time

        defender_velocity = np.linalg.norm(
            [delta_x[closest_defender_index], delta_y[closest_defender_index]]) / delta_time


        # Append the shot_features1 to the list
        features.append((frame, shooter, shooter_name, shooter_offensive_rating, dist, shooter_coords[0],
                         shooter_coords[1], shot_angle, shooter_velocity, fg_percentage_zone, closest_defender,
                         closest_defender_name, defender_defensive_rating, closest_defender_dist, shot_defender_angle,
                         defender_velocity, num_close_defenders, shot_clock, score_margin, quarter, minutes, seconds,
                         points_attempted, make))

        if not filtered_zones['SHOT_ZONE_BASIC'].empty:
            zone.append((shooter_coords[0], shooter_coords[1], dist, filtered_zones['SHOT_ZONE_BASIC'].values[0],
                        filtered_zones['SHOT_ZONE_AREA'].values[0], filtered_zones['SHOT_ZONE_RANGE'].values[0]))

        # Write shot_features as csv file
        # if write_file:
        #     with open("features/{filename}".format(filename=filename), 'a') as myfile:
        #         myfile.write("{frame}, {basket}, {shooter}, {shooter_name}, {shooter_offensive_rating}, {dist}, {x}, {y}, "
        #                      "{shot_angle}, {shooter_velocity}, {fg_percentage_zone}, {closest_defender}, "
        #                      "{closest_defender_name}, {defender_defensive_rating}, {closest_defender_dist}, "
        #                      "{shot_defender_angle}, {defender_velocity}, {num_close_defenders}, {shot_clock}, "
        #                      "{score_margin}, {quarter}, {minutes}, {seconds}, {points_attempted}, {make}\n"
        #                      .format(frame=frame, basket=basket, shooter=shooter, shooter_name=shooter_name,
        #                              shooter_offensive_rating=shooter_offensive_rating, dist=dist,
        #                              x=shooter_coords[0], y=shooter_coords[1], shot_angle=shot_angle,
        #                              shooter_velocity=shooter_velocity, fg_percentage_zone=fg_percentage_zone,
        #                              closest_defender=closest_defender, closest_defender_name=closest_defender_name,
        #                              defender_defensive_rating=defender_defensive_rating,
        #                              closest_defender_dist=closest_defender_dist, shot_defender_angle=shot_defender_angle,
        #                              defender_velocity=defender_velocity, num_close_defenders=num_close_defenders,
        #                              shot_clock=shot_clock, score_margin=score_margin, quarter=quarter, minutes=minutes,
        #                              seconds=seconds, points_attempted=points_attempted, make=make))


    # Write the features to a pickle file
    with open("./sfeatures/{filename}".format(filename=filename), 'wb') as myfile:
        pickle.dump(features, myfile)

    with open("./zones/{filename}".format(filename=filename), 'wb') as myfile:
        pickle.dump(zone, myfile)

    print("Features extracted for {filename}".format(filename=filename))

    return features


def write_features(gamelist):
    """
    Writes all spacing statistics to data/spacing directory for each game
    """

    # Write all the labels in a text file
    with open('labels.txt', 'w') as myfile:
        myfile.write("frame, shooter, shooter_name, shooter_offensive_rating, dist, x, y, shot_angle, "
                     "shooter_velocity, fg_percentage_zone, closest_defender, closest_defender_name, "
                     "closest_defender_defensive_rating, closest_defender_dist, closest_defender_angle, "
                     "closest_defender_velocity, num_close_defenders, shot_clock, score_margin, quarter, 'minutes', "
                     "'seconds', 'points_attempted', make\n")

    for game in gamelist:
        try:
            get_features(game[0], game[1], game[2],
                         write_file=True, write_score=False)
        except:
            with open('errorlog.txt', 'a') as myfile:
                myfile.write("{game} Could not extract spacing data\n"
                             .format(game=game))


write_features(extract_games())
