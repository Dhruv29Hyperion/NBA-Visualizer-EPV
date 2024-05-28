# NBA VISUALIZER TOOLKIT

## Project Overview
This project aims to optimize basketball offense by developing an extensive toolkit for coaches, players, and administrators. It focuses on analyzing basketball games and offensive schemes to enhance team performance and strategy formulation.

### Course
End Semester Project - Machine Learning for Pattern Recognition

## Authors
Dhruv Srivastava and Yash Sangtani

## Sections

### 1. Introduction
- Overview of basketball and the project's aim to quantify the value of each player's future actions based on in-game scenarios.

### 2. Problem Statement
- Enhance team performance and strategy formulation by determining the decision with the highest probability of success during an offensive possession.

### 3. Literature Survey
- Review of prior research on using player tracking data, deep learning, and machine learning to analyze basketball performance and strategies.

### 4. Datasets and Preprocessing
#### Primary Dataset
- Tracking data from SportsVU, including comprehensive snapshots of the basketball court captured by six cameras.

#### Secondary Dataset
- Play-by-play event data from NBA games, detailing game events, player actions, and game scores.

#### Ratings (NBA_API)
- Integration with game logs to extract meaningful events and create player-specific stats.

### 5. Data and Feature Preprocessing
- Extracting, parsing, and interpreting large JSON data.
- Filtering out possessions, passes, and shot attempts.
- Calculating distance and angle features, and finding player ratings.

### 6. Pass Features
- Various features such as distance, angle, and defender proximity used to evaluate pass difficulty.

### 7. Pass Difficulty Model
- Specifications and evaluation of the pass difficulty model using a neural network with regularization and dropout.

### 8. Shot Features
- Various features such as shooter rating, shot coordinates, and defender velocity used to evaluate shot difficulty.

### 9. Shot Difficulty Model
- Specifications and evaluation of the shot difficulty model using a neural network with multiple layers.

### 10. Expected Possession Value (EPV)
- Aggregating EPV values to compare expected and actual points scored.

### 11. Deployability and Future Prospects
- Using the model for in-depth analysis by coaches and extending it to team-scale evaluation of strategies and tactics.

### 12. Game Visualizer (with EPV)
- Integration of EPV in a visualizer to analyze player performance and decision-making during games.

### 13. Literature Review for Future Work
- Emphasizing the potential of machine learning in sports analytics and optimizing strategies in other sports.

---

This README provides a concise summary of the project "Basketball Offense Optimization" and outlines the structure and key components. For detailed analysis and implementation, please refer to the full project document.
