import matplotlib
import os
import subprocess
import warnings
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from subprocess import Popen, PIPE
import numpy as np
import seaborn as sns
from scipy.spatial import ConvexHull
from matplotlib.patches import Rectangle, Circle, Arc, Polygon
from shot_tracking_plot import extract_shot_probability

matplotlib.use('TkAgg')

# Initialize project
if not os.path.exists('temp'):
    os.mkdir('temp')

# Suppress warnings
warnings.filterwarnings("ignore")


class Game(object):
    """
    Class for basketball game.
    Contains play by play and player tracking data and methods for
    analysis and plotting.
    """

    def __init__(self, date, team1, team2):
        """
        Args:
            date (str): 'MM.DD.YYYY', date of game
            team1 (str): 'XXX', abbreviation of team1 in data
                tracking file name
            team2 (str): 'XXX', abbreviation of team2 in data
                tracking file name

        Attributes:
            date (str): 'MM.DD.YYYY', date of game
            team1 (str): 'XXX', abbreviation of team1 in data
                tracking file name
            team2 (str): 'XXX', abbreviation of team2 in data
                tracking file name
            tracking_id (str): id to access player tracking data
                Due to the way the SportVU data is stored, game_id is
                complicated: 'MM.DD.YYYY.AWAYTEAM.at.HOMETEAM'
                For Example: 01.13.2016.GSW.at.DEN
            tracking_data (dict): Dictionary of unstructured tracking
                data scraped from GitHub.
            game_id (str): ID for game. Luckily, SportVU and play by
                play use the same game ID
            pbp (pd.DataFrame): Play by play data.  33 columns per pbp
                instance.
            moments (pd.DataFrame): DataFrame of player tracking data.
                Each entry is a single snapshot of where the players
                are at a given time on the court.
                Columns: ['quarter', 'universe_time', 'quarter_time',
                'shot_clock', 'positions', 'game_time'].
                moments['positions'] contains a list of where each player
                and the ball are located.
            player_ids (dict): dictionary of {player: player_id} for
                all players in game.
            away_id (int): ID of away team
            home_id (int): ID of home team
            team_colors (dict): dictionary of colors for each team and
                ball. Used for plotting.
            home_team (str): 'XXX', abbreviation of home team
            away_team (str): 'XXX', abbreviation of the away team
        """

        self.date = date
        self.team1 = team1
        self.team2 = team2
        self.flip_direction = False
        self.tracking_id = f'{self.date}.{self.team2}.at.{self.team1}'
        self.tracking_data = None
        self.game_id = None
        self.pbp = None
        self.epv = None
        self.moments = None
        self.player_ids = None
        self.jersey_numbers = None
        self._get_tracking_data()
        self._get_playbyplay_data()
        self._format_tracking_data()
        self._get_player_ids()
        self._get_jersey_numbers()
        self.away_id = self.tracking_data['events'][0]['visitor']['teamid']
        self.home_id = self.tracking_data['events'][0]['home']['teamid']
        self.team_colors = {-1: "orange",
                            self.away_id: "blue",
                            self.home_id: "red"}
        self.home_team = (self.tracking_data['events'][0]['home']
        ['abbreviation'])
        self.away_team = (self.tracking_data['events'][0]['visitor']
        ['abbreviation'])
        self.flip_direction = False
        self._determine_direction()
        print('All data is loaded')

    def _get_tracking_data(self):
        """
        Retrieves tracking data from basketballPlay/Optimisation/Tracking_Data/'game logs' folder
        The tracking data is in .7z format with in 'date.team1.at.team2' format. Unzip it and store it in ./temp folder
        """

        tracking_folder = 'C:/Users/Dhruv/PycharmProjects/basketballPlayOptimisation/Tracking_Data/game logs'
        tracking_file_path = os.path.normpath(os.path.join(tracking_folder, f'{self.tracking_id}.7z'))
        subprocess.run(['7z', 'e', tracking_file_path, '-o./temp'], shell=True)

        # Extract game ID from extracted file name.
        for file in os.listdir('./temp'):
            if os.path.splitext(file)[1] == '.json':
                self.game_id = file[:-5]

        # Load tracking data and remove json file
        with open(f'./temp/{self.game_id}.json') as data_file:
            self.tracking_data = json.load(data_file)  # Load this json
        os.remove(f'./temp/{self.game_id}.json')
        return self

    def _get_playbyplay_data(self):
        """
        Retrieves play by play data from basketballPlay/Optimisation/Tracking_Data/'events' folder
        The play by play data is in in game_id.csv format
        """

        pbp_folder = 'C:/Users/Dhruv/PycharmProjects/basketballPlayOptimisation/Tracking_Data/events'
        pbp_file_path = os.path.normpath(os.path.join(pbp_folder, f'{self.game_id}.csv'))
        pbp = pd.read_csv(pbp_file_path)
        # Store the play by play data in a dataframe
        self.pbp = pd.DataFrame(pbp)

        # Get time in quarter remaining to cross-reference tracking data
        self.pbp['Qmin'] = (self.pbp['PCTIMESTRING'].str.split(':', expand=True)[0])
        self.pbp['Qsec'] = (self.pbp['PCTIMESTRING'].str.split(':', expand=True)[1])
        self.pbp['Qtime'] = (self.pbp['Qmin'].astype(int) * 60 + self.pbp['Qsec'].astype(int))
        self.pbp['game_time'] = ((self.pbp['PERIOD'] - 1) * 720 + (720 - self.pbp['Qtime']))

        # Format score so that it makes sense: 'XX-XX'
        self.pbp['SCORE'] = (self.pbp['SCORE'].fillna(method='ffill').fillna('0 - 0'))
        return self

    def _get_player_ids(self):
        """
        Helper function for returning player ids for all players in game.
        Note: This data may also be somewhere more conveniently
            accessible in tracking_data.
        """
        ids = {}
        for player in self.tracking_data['events'][0]['home']['players']:
            ids[player['firstname'] + ' ' + player['lastname']] = player['playerid']

        for player in self.tracking_data['events'][0]['visitor']['players']:
            ids[player['firstname'] + ' ' + player['lastname']] = player['playerid']

        self.player_ids = ids

    def _get_jersey_numbers(self):
        """
        Helper function for returning player jersey numbers for all players in game.
        This is useful for identifying players by their jersey numbers.
        """

        home_jersey_numbers = {}
        for player in self.tracking_data['events'][0]['home']['players']:
            home_jersey_numbers[player['firstname'] + ' ' + player['lastname']] = player['jersey']

        away_jersey_numbers = {}
        for player in self.tracking_data['events'][0]['visitor']['players']:
            away_jersey_numbers[player['firstname'] + ' ' + player['lastname']] = player['jersey']

        self.jersey_numbers = {'home': home_jersey_numbers, 'away': away_jersey_numbers}

    def _format_tracking_data(self):
        """
        Helper function to format tracking data into pandas DataFrame
        """
        events = pd.DataFrame(self.tracking_data['events'])
        moments = []

        # Extract 'moments': Each moment is an individual frame
        for row in events['moments']:
            for inner_row in row:
                moments.append(inner_row)
        moments = pd.DataFrame(moments)
        moments = moments.drop_duplicates(subset=[1])
        moments = moments.reset_index()

        moments.columns = ['index', 'quarter', 'universe_time', 'quarter_time',
                           'shot_clock', 'unknown', 'positions']
        moments['game_time'] = (moments.quarter - 1) * 720 + \
                               (720 - moments.quarter_time)
        moments.drop(['index', 'unknown'], axis=1, inplace=True)
        self.moments = moments
        return self

    def _draw_court(self, color="gray", lw=2, grid=False, zorder=0):
        """
        Helper function to draw court.
        """
        ax = plt.gca()

        # Create the court lines
        outer = Rectangle((0, -50), width=94, height=50, color="#f0e6d2",
                          zorder=zorder, fill=True, lw=lw)
        l_hoop = Circle((5.35, -25), radius=.75, lw=lw, fill=False,
                        color=color, zorder=zorder)
        r_hoop = Circle((88.65, -25), radius=.75, lw=lw, fill=False,
                        color=color, zorder=zorder)
        l_backboard = Rectangle((4, -28), 0, 6, lw=lw, color=color,
                                zorder=zorder)
        r_backboard = Rectangle((90, -28), 0, 6, lw=lw, color=color,
                                zorder=zorder)
        l_outer_box = Rectangle((0, -33), 19, 16, lw=lw, fill=False,
                                color=color, zorder=zorder)
        l_inner_box = Rectangle((0, -31), 19, 12, lw=lw, fill=False,
                                color=color, zorder=zorder)
        r_outer_box = Rectangle((75, -33), 19, 16, lw=lw, fill=False,
                                color=color, zorder=zorder)
        r_inner_box = Rectangle((75, -31), 19, 12, lw=lw, fill=False,
                                color=color, zorder=zorder)
        l_free_throw = Circle((19, -25), radius=6, lw=lw, fill=False,
                              color=color, zorder=zorder)
        r_free_throw = Circle((75, -25), radius=6, lw=lw, fill=False,
                              color=color, zorder=zorder)
        l_corner_a = Rectangle((0, -3), 14, 0, lw=lw, color=color,
                               zorder=zorder)
        l_corner_b = Rectangle((0, -47), 14, 0, lw=lw, color=color,
                               zorder=zorder)
        r_corner_a = Rectangle((80, -3), 14, 0, lw=lw, color=color,
                               zorder=zorder)
        r_corner_b = Rectangle((80, -47), 14, 0, lw=lw, color=color,
                               zorder=zorder)
        l_arc = Arc((5, -25), 47.5, 47.5, theta1=292, theta2=68, lw=lw,
                    color=color, zorder=zorder)
        r_arc = Arc((89, -25), 47.5, 47.5, theta1=112, theta2=248,
                    lw=lw, color=color, zorder=zorder)
        half_court = Rectangle((47, -50), 0, 50, lw=lw, color=color,
                               zorder=zorder)
        hc_big_circle = Circle((47, -25), radius=6, lw=lw, fill=False,
                               color=color, zorder=zorder)
        hc_sm_circle = Circle((47, -25), radius=2, lw=lw, fill=False,
                              color=color, zorder=zorder)
        court_elements = [l_hoop, l_backboard, l_outer_box, outer,
                          l_inner_box, l_free_throw, l_corner_a,
                          l_corner_b, l_arc, r_hoop, r_backboard,
                          r_outer_box, r_inner_box, r_free_throw,
                          r_corner_a, r_corner_b, r_arc, half_court,
                          hc_big_circle, hc_sm_circle]

        # Add the court elements onto the axes
        for element in court_elements:
            ax.add_patch(element)

        return ax

    def animate_play(self, game_time, length, highlight_player=None,
                     commentary=True, show_spacing=None):
        """
        Method for animating plays in game.
        Outputs video file of play in {cwd}/temp.
        Individual frames are streamed directly to ffmpeg without writing them
        to the disk, which is a great speed improvement over watch_play

        Args:
            game_time (int): time in game to start video
                (seconds into the game).
                Currently, game_time can also be a tuple of length two
                with (starting_frame, ending_frame)if you want to
                watch a play using frames instead of game time.
            length (int): length of play to watch (seconds)
            highlight_player (str): If not None, video will highlight
                the circle of the inputted player for easy tracking.
            commentary (bool): Whether to include play-by-play commentary in
                the animation
            show_spacing (str) in ['home', 'away']: show convex hull
                spacing of a home or away team.
                If None, does not show spacing.

        Returns: an instance of self, and outputs video file of play
        """

        if type(game_time) == tuple:
            starting_frame = game_time[0]
            ending_frame = game_time[1]
        else:
            # Get starting and ending frame from requested game_time and length
            starting_frame = self.moments[self.moments.game_time.round() ==
                                          game_time].index.values[0]
            ending_frame = self.moments[self.moments.game_time.round() ==
                                        game_time + length].index.values[0] + 6

        # Make video of each frame
        filename = "./temp/{game_time}.mpeg".format(game_time=game_time)
        if commentary:
            size = (960, 960)
        else:
            size = (960, 480)
        cmdstring = (
            'ffmpeg',
            '-y',  # Overwrite output files without asking
            '-framerate', '20',  # Set input frame rate
            '-i', 'temp/%04d.png',  # Input file pattern
            '-pix_fmt', 'yuv420p',  # Set pixel format
            '-crf', '0',  # Set the constant rate factor
            filename,
        )

        for frame in range(starting_frame, ending_frame):
            self.plot_frame(frame, highlight_player=highlight_player,
                            commentary=commentary, show_spacing=show_spacing)

        #Execute the command
        subprocess.run(cmdstring, shell=True)

        # Remove all the frames
        for file in os.listdir('temp'):
            if file.endswith('.png'):
                os.remove('temp/' + file)

        return self

    def watch_player_actions(self, player_name, action, length=15, max_vids=5):
        """
        Method for viewing all plays a player in the game had of a
        specified type.
        For example: all of Damian Lillards FG attempts in the game
        Outputs video file for each play in {cwd}/temp

        Args:
            player_name (str): Name of player for which to produce videos.
                Currently, player_name must be perfectly formatted and
                capitalized, since no string processing is performed.
            action (str) {'all_FG', 'made_FG', 'miss_FG', 'rebound'}:
                Action type of interest
            length (int): length of play to watch (seconds) for each action.
            max_vids (int): Maximum number of videos to produce.
                max_vids=None if all videos are desired.  If max_vids
                is less than the total number of actions in the game, the
                earliest actions are made into videos.

        Returns: an instance of self, and outputs video file of plays
        """
        player_action_times = self._get_player_actions(player_name, action)
        for index, time in enumerate(player_action_times):
            if index == max_vids:
                break
            self.animate_play(time - length, length,
                              highlight_player=player_name,
                              commentary=False)
        return self

    def _get_commentary(self, game_time, commentary_length=6,
                        commentary_depth=10):
        """
        Helper function for returning play by play events for a
        given game time.

        Args:
            game_time (int): game time (in seconds) for which to
                retrieve commentary for
            commentary_length (int): Number of play-by-play calls to
                include in commentary
            commentary_depth (int): Number of seconds to look in past
                to retrieve play-by-play calls
                commentary_depth=10 looks at previous 10 seconds of
                game for play-by-play calls

        Returns: tuple of information (commentary_script, score)
            commentary_script (str): string of commentary
                Most recent pl ay-by-play calls, seperated by line breaks
            score (str): Score at current time 'XX - XX'
        """
        commentary = [' ' for i in range(commentary_length)]
        commentary[0] = '.'
        count = 0
        score = "0 - 0"
        for game_second in range(game_time - commentary_depth, game_time + 2):
            for index, row in self.pbp[self.pbp.game_time ==
                                       game_second].iterrows():
                if count >= commentary_length - 1:
                    break
                if row['HOMEDESCRIPTION']:
                    commentary[count] = ('{self.home_team}: '
                                         .format(self=self) +
                                         str(row['HOMEDESCRIPTION']))
                    count += 1
                if row['VISITORDESCRIPTION']:
                    commentary[count] = ('{self.away_team}: '
                                         .format(self=self) +
                                         str(row['VISITORDESCRIPTION']))
                    count += 1
                if row['NEUTRALDESCRIPTION']:
                    commentary[count] = str(row['NEUTRALDESCRIPTION'])
                    count += 1
                score = str(row['SCORE'])

        # Remove nan values from commentary
        for i in range(commentary_length):
            if commentary[i] == 'nan' or commentary[i] == '{self.home_team}: nan'.format(self=self) or commentary[i] == '{self.away_team}: nan'.format(self=self):
                commentary[i] = ''

        # Join the commentary into multiple lines if it isn't ''
        commentary_script = '\n'.join([comment for comment in commentary if comment != ''])

        return commentary_script, score

    def _get_player_actions(self, player_name, action):
        """
        Helper function to get all times a player performed a specific action

        Args:
            player_name (str): name of player to get all actions for
            action {'all_FG', 'made_FG', 'miss_FG', 'rebound'}:
                Type of action to get all times for.

        Returns:
            times (list): a list of game times a player performed a
                specific action
        """
        player_id = self.player_ids[player_name]
        action_dict = {'all_FG': [1, 2], 'made_FG': [1],
                       'miss_FG': [2], 'rebound': [4]}
        action_df = self.pbp[(self.pbp['PLAYER1_ID'] == player_id) &
                             (self.pbp['EVENTMSGTYPE']
                              .isin(action_dict[action]))]
        times = list(action_df['game_time'])
        return times

    def _get_moment_details(self, frame_number, highlight_player=None):
        """
        Helper function for getting important information for a given frame

        Args:
            frame_number (int): Frame in game to retrieve data for
                frame_number gets' the player tracking data from
                    moments.iloc[frame_number]
            highlight_player (str): Name of player to be highlighted
                in downstream plotting.
                if None, no player is highlighted.

        Returns: tuple of data
            game_time (int): seconds into game of current moment
            x_pos (list): list of x coordinants for all players and ball
            y_pos (list): list of y coordinants for all players and ball
            colors (list): color coding of each player/ball for coordinant data
            sizes (list): size of each player/ball
                (used for showing ball height)
            quarter (int): Game quarter
            shot_clock (str): shot clock
            game_clock (str): game clock
            edges (list): list of marker edge sizes of each player for video.
                useful when trying to highlight a player by making
                their edge thicker.
            universe_time (int): Time in the universe, in msec
        """
        current_moment = self.moments.iloc[frame_number]
        game_time = int(np.round(current_moment['game_time']))
        universe_time = int(current_moment['universe_time'])
        x_pos, y_pos, colors, sizes, edges, ids = [], [], [], [], [], []
        # Get player positions
        for player in current_moment.positions:
            x_pos.append(player[2])
            y_pos.append(player[3])
            colors.append(self.team_colors[player[0]])
            # Use ball height for size (useful to sevie a shot)
            if player[0] == -1:
                sizes.append(max(150 - 2 * (player[4] - 5) ** 2, 10))
            else:
                sizes.append(200)
            # highlight_player makes their outline much thicker on the video
            if (highlight_player and
                    player[1] == self.player_ids[highlight_player]):
                edges.append(5)
            else:
                edges.append(0.5)

            ids.append(player[1])

        # Unfortunately, the plot is below the y-axis,
        # so the y positions need to be corrected
        y_pos = np.array(y_pos) - 50
        shot_clock = current_moment.shot_clock
        if np.isnan(shot_clock):
            shot_clock = 24.00
        shot_clock = str(shot_clock).split('.')[0]
        game_min, game_sec = divmod(current_moment.quarter_time, 60)
        game_clock = "%02d:%02d" % (game_min, game_sec)
        quarter = current_moment.quarter
        return (game_time, x_pos, y_pos, colors, sizes, quarter,
                shot_clock, game_clock, edges, universe_time, ids)

    def plot_frame(self, frame_number, highlight_player=None,
                   commentary=True, show_spacing=False, show_epv=False,
                   plot_spacing=False, pipe=None):
        """
        Creates an individual the frame of game.
        Outputs .png file in {cwd}/temp

        Args:
            frame_number (int): number of frame in game to create
                frame_number gets player tracking data from
                moments.iloc[frame_number]
            highlight_player (str): Name of player to highlight
                (by making their outline thicker).
                if None, no player is highlighted
            commentary (bool): if True, add play-by-play commentary
                under frame
            show_spacing (str in ['home', 'away']): show convex hull
                of home or away team
                if None, does not display any convex hull
            pipe (subprocesses.Popen): Popen object with open pipe
                to send image to if False, image is written to disk
                instead of sent to pipe
            show_epv (bool): if True, add epv values of each offensive player
                on top of their coordinate

        Returns: an instance of self, and outputs .png file of frame
            If pipe, ARGB values are sent to pipe object instead of
            writing to disk.
        """
        (game_time, x_pos, y_pos, colors, sizes,
         quarter, shot_clock, game_clock, edges,
         universe_time, ids) = self._get_moment_details(frame_number,
                                                        highlight_player=highlight_player)

        if show_epv:
            self.epv = extract_shot_probability(self, self._get_moment_details(frame_number),
                                                self._get_moment_details(frame_number - 1))

        (commentary_script, score) = self._get_commentary(game_time)
        fig = plt.figure()
        self._draw_court()
        frame = plt.gca()
        frame.axes.get_xaxis().set_ticks([])
        frame.axes.get_yaxis().set_ticks([])

        # Plot players and ball and embed the jersey number of the player after getting the name of the player and
        # then matching it with the jersey number

        for x, y, color, size, edge, player_id in zip(x_pos, y_pos, colors, sizes, edges, ids):
            if player_id in self.player_ids.values():
                # Get the player epv value for the player id using self.epv dictionary, if it isn't empty
                # Plot on top of the player's coordinate
                if show_epv and self.epv:
                    if player_id in self.epv.keys():
                        epv = self.epv[player_id]
                        plt.annotate(f'{epv:.2f}', xy=(x+3, y+3), fontsize=10, ha='center', va='top', color='black')
                player_name = [k for k, v in self.player_ids.items() if v == player_id][0]
                jersey_number = self.jersey_numbers['home'].get(player_name, None)
                if jersey_number is None:
                    jersey_number = self.jersey_numbers['away'].get(player_name, None)
                if jersey_number is not None:
                    plt.annotate(jersey_number, xy=(x, y), fontsize=12, ha='center', va='center', color='white')

            plt.scatter(x, y, s=size, c=color, edgecolors='black', linewidth=edge)

        plt.xlim(-5, 100)
        plt.ylim(-55, 5)
        sns.set_style('dark')

        if commentary:
            plt.figtext(0.15, -.20, commentary_script, size=20)
        plt.figtext(0.43, 0.125, shot_clock, size=18)
        plt.figtext(0.5, 0.125, 'Q' + str(quarter), size=18)
        plt.figtext(0.57, 0.125, str(game_clock), size=18)
        plt.figtext(0.37, .83,
                    self.away_team + "  " + score + "  " + self.home_team,
                    size=18)
        if highlight_player:
            plt.figtext(0.17, 0.85, highlight_player, size=18)

        # Add team color indicators to top of frame
        plt.scatter([25, 75], [2.5, 2.5], s=100,
                    c=[self.team_colors[self.away_id],
                       self.team_colors[self.home_id]])

        # Side panel for player names and jersey numbers
        side_ax_home = fig.add_axes([-0.01, 0.1, 0.1, 0.8], frameon=False)
        side_ax_home.axes.get_xaxis().set_ticks([])
        side_ax_home.axes.get_yaxis().set_ticks([])
        side_ax_home.set_xlim(0, 1)

        side_ax_away = fig.add_axes([0.95, 0.1, 0.1, 0.8], frameon=False)
        side_ax_away.axes.get_xaxis().set_ticks([])
        side_ax_away.axes.get_yaxis().set_ticks([])
        side_ax_away.set_xlim(0, 1)

        # Add only the players that are on the court
        on_court_players_home = [k for k, v in self.player_ids.items() if v in ids and k in self.jersey_numbers['away']]
        on_court_players_away = [k for k, v in self.player_ids.items() if v in ids and k in self.jersey_numbers['home']]

        side_ax_home.set_ylim(0, len(on_court_players_home) * 0.05 + 0.1)  # Adjusted ylim
        side_ax_away.set_ylim(0, len(on_court_players_away) * 0.05 + 0.1)  # Adjusted ylim

        for i, player in enumerate(on_court_players_home):
            side_ax_home.text(0.4, 0.6 - i * 0.05, player + " - " + str(self.jersey_numbers['away'][player]), size=8,
                              ha='center', va='center', transform=side_ax_home.transAxes)

        for i, player in enumerate(on_court_players_away):
            side_ax_away.text(0.4, 0.6 - i * 0.05, player + " - " + str(self.jersey_numbers['home'][player]), size=8,
                              ha='center', va='center', transform=side_ax_away.transAxes)

        if show_spacing:
            # Show convex hull on frame
            xy_pos = np.column_stack((np.array(x_pos), np.array(y_pos)))
            if show_spacing == 'home':
                points = xy_pos[1:6, :]
            if show_spacing == 'away':
                points = xy_pos[6:, :]
            hull = ConvexHull(points)
            hull_points = points[hull.vertices, :]
            polygon = Polygon(hull_points, alpha=0.3, color='gray')
            # ax = plt.gca()
            frame.add_patch(polygon)
        if pipe:
            # Write ARGB values to pipe
            fig.canvas.draw()
            string = fig.canvas.tostring_argb()
            pipe.stdin.write(string)
            plt.close()
            if commentary:
                fig = plt.figure(figsize=(12, 6))
                plt.figtext(.2, .4, commentary_script, size=20)
                fig.canvas.draw()
                string = fig.canvas.tostring_argb()
                pipe.stdin.write(string)
            plt.close()

        else:
            # Save image to disk
            plt.savefig('temp/{frame_number}.png'
                        .format(frame_number=frame_number),
                        bbox_inches='tight')
            plt.close()
        return self

    def _in_formation(self, frame_number):
        """
        This is a complicated method to explain, but it is actually
        very simple.
        It determines if the game is in a set offense/defense.
        It basically returns True if a normal play is being run,
        and False if the game is in transition, out of bounds,
        free throw, etc. It is useful for analyzing plays that teams
        run, and discarding all extranous times from the game.
        """
        # Get relevant moment details
        details = self._get_moment_details(frame_number)
        x_pos = np.array(details[1])
        shot_clock = details[6]
        # Determine if offense/defense is set
        if float(shot_clock) < 23:
            if (x_pos < 47).all() or (x_pos > 47).all():
                return True
        return False

    def get_spacing_area(self, frame_number):
        """
        Calculates convex hull of home and away team for a given frame.
        Useful for analyzing the spacing of teams.

        Args:
            frame_number (int): number of frame in game to calculate
                team convex hulls

        Returns: tuple of data (home_area, away_area)
            home_area (float): convex hull area of home team
            away_area (float): convex hull area of away team

        """
        details = self._get_moment_details(frame_number)
        x_pos = np.array(details[1])
        y_pos = np.array(details[2])
        xy_pos = np.column_stack((x_pos, y_pos))
        home_area = ConvexHull(xy_pos[1:6, :]).area
        away_area = ConvexHull(xy_pos[6:, :]).area
        return (home_area, away_area)

    def get_offensive_team(self, frame_number):
        """
        Determines which team is on offense.
        Currently only works if team is in set offense or defense.

        Args:
            frame_number (int): number of frame in game to determine
                offensive team

        Returns:
            str in ['home', 'away']
        """
        details = self._get_moment_details(frame_number)
        x_pos = np.array(details[1])
        quarter = details[5]
        if len(x_pos) != 11:
            return None
        if self.flip_direction:
            if (x_pos < 47).all() and quarter in [1, 2]:
                return 'away'
            if (x_pos > 47).all() and quarter in [3, 4]:
                return 'away'
            if (x_pos < 47).all() and quarter in [3, 4]:
                return 'home'
            if (x_pos > 47).all() and quarter in [1, 2]:
                return 'home'
        if (x_pos < 47).all() and quarter in [1, 2]:
            return 'home'
        if (x_pos > 47).all() and quarter in [3, 4]:
            return 'home'
        if (x_pos < 47).all() and quarter in [3, 4]:
            return 'away'
        if (x_pos > 47).all() and quarter in [1, 2]:
            return 'away'
        return None

    def _determine_direction(self):
        """
        Helper funcation to determine which direction the home team is going.
        Surprisingly, this is not consistent and depends on the game.
        Currently, this method detects which side the players start on and is
        ~90% accurate
        """
        incorrect_count = 0
        correct_count = 0
        for frame in range(0, 10000, 100):
            details = self._get_moment_details(frame)
            home_team_x = details[1][1:6]
            away_team_x = details[1][6:]
            if np.mean(home_team_x) < np.mean(away_team_x):
                incorrect_count += 1
            else:
                correct_count += 1
        if incorrect_count > correct_count:
            self.flip_direction = True
        return None

    def get_frame(self, game_time):
        """
        Converts a game time to a frame number.  Useful all over the place.

        Args:
            game_time (int): game time in seconds of interest

        Returns:
            frame (int): frame number of game time
        """
        test_time = game_time
        frame = 0
        while True and test_time > 0:
            if test_time in self.moments.game_time.round():
                frames = self.moments[self.moments.game_time.round() ==
                                      test_time].index.values
                if len(frames) > 0:
                    frame = frames[0]
                    break
                else:
                    test_time -= 1
            else:
                test_time -= 1
        return frame

    def get_play_frames(self, event_num, play_type='offense'):
        """
        Args:
            event_num (int): EVENTNUM of interest in games.pbp
                NOTE: Check pbpevents.txt for event numbers
            play_type (str in ['offense', 'defense']): Team of interest
                is offense or defense

        Returns:
            tuple of (start_time (int), end_time (int)): start time
                and end time in seconds for play of interest
        """
        play_index = self.pbp[self.pbp['EVENTNUM'] == event_num].index[0]
        event_team = str(self.pbp[self.pbp['EVENTNUM'] == event_num]
                         .PLAYER1_TEAM_ABBREVIATION.head(1).values[0])
        if event_team == self.home_team:
            target_team = 'home'
        if event_team == self.away_team:
            target_team = 'away'
        end_time = int(self.pbp[self.pbp['EVENTNUM'] == event_num].game_time)
        # To find lower bound on the starting frame of the play,
        # determining when previous play ended
        putative_start_time = int(self.pbp.iloc[play_index - 1].game_time)
        putative_start_frame = self.get_frame(putative_start_time)
        end_frame = self.get_frame(end_time)
        # for test_frame in range(putative_start_frame, end_frame):
        #     if self.get_offensive_team(test_frame) == target_team:
        #         break
        # # If the previous loop never found an offensive play,
        # # the function returns None
        # else:
        #     return None
        # Subtract 2 seconds to get the start of the shot
        start_frame = self.get_frame(round(self.moments.iloc[putative_start_frame].game_time - 2))
        return start_frame, end_frame

    def watch_play(self, game_time, length, highlight_player=None,
                   commentary=True, show_spacing=None, show_epv=False):
        """
        DEPRECIATED.  See animate_play() for similar (fastere) method

        Method for viewing plays in game.
        Outputs video file of play in {cwd}/temp

        Args:
            game_time (int): time in game to start video
                (seconds into the game).
                Currently game_time can also be an tuple of length
                two with (starting_frame, ending_frame) if you want
                to watch a play using frames instead of game time.
            length (int): length of play to watch (seconds)
            highlight_player (str): If not None, video will highlight
                the circle of the inputed player for easy tracking.
            commentary (bool): Whether to include play-by-play
                commentary underneath video
            show_spacing (str in ['home', 'away']): show convex hull
                of home or away team.
                if None, does not display any convex hull

        Returns: an instance of self, and outputs video file of play
        """

        if type(game_time) == tuple:
            starting_frame = game_time[0]
            ending_frame = game_time[1]
        else:
            # Get starting and ending frame from requested game_time and length
            starting_frame = self.moments[self.moments.game_time.round() ==
                                          game_time].index.values[0]
            ending_frame = self.moments[self.moments.game_time.round() ==
                                        game_time + length].index.values[0]

        # Make video of each frame
        for frame in range(starting_frame, ending_frame):
            self.plot_frame(frame, highlight_player=highlight_player,
                            commentary=commentary, show_spacing=show_spacing, show_epv=show_epv)
        command = ('ffmpeg -framerate 10 -start_number {starting_frame} '
                   '-i %d.png -c:v libx264 -r 30 -pix_fmt yuv420p -vf '
                   '"scale=trunc(iw/2)*2:trunc(ih/2)*2" {starting_frame}'
                   '.mp4').format(starting_frame=starting_frame)
        os.chdir('temp')
        os.system(command)
        os.chdir('..')

        # Delete images
        for file in os.listdir('./temp'):
            if os.path.splitext(file)[1] == '.png':
                os.remove('./temp/{file}'.format(file=file))

        return self
