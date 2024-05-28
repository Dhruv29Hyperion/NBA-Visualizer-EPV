import pandas as pd

# Load the player_ratings.csv file
player_ratings = pd.read_csv('player_rating.csv')

# There are some rows with the same 'PLAYER_NAME'
# average out the ratings for these players and keep only one row

player_ratings = player_ratings.groupby('PLAYER_NAME').mean().reset_index()

# Save the cleaned data to a new file
player_ratings.to_csv('player_ratings.csv', index=False)
