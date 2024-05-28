import pandas as pd

# Sample data
data = pd.read_csv("shots.csv")  # Replace "your_file.csv" with your file path

# Group by player, shot zone basic, shot zone area, shot zone range, and shot distance
grouped_data = data.groupby(['PLAYER_NAME', 'SHOT_ZONE_BASIC', 'SHOT_ZONE_AREA', 'SHOT_ZONE_RANGE'])

# Calculate shot attempts and made shots for each player and shot factors
shot_stats = grouped_data['SHOT_MADE_FLAG'].agg(['count', 'sum'])

# Calculate shot percentage
shot_stats['SHOT_PERCENTAGE'] = (shot_stats['sum'] / shot_stats['count']) * 100

# Reset index to make PLAYER_NAME and other factors as columns
shot_stats.reset_index(inplace=True)

# Pivot the data to have each shot factor as columns
shot_percentage_data = pd.pivot_table(
    shot_stats,
    values='SHOT_PERCENTAGE',
    index=['PLAYER_NAME', 'SHOT_ZONE_BASIC', 'SHOT_ZONE_AREA', 'SHOT_ZONE_RANGE'],
)

# Fill NaN values with 0, meaning no attempts in that zone
shot_percentage_data.fillna(0, inplace=True)

# Optionally, you can export the dataset to a file
shot_percentage_data.to_csv("player_shot_percentages.csv")
