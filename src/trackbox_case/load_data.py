import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from trackbox_case import feature_engineering
import re

def load_match_data():

    #obtain all the separated home and away player data per match in this dictionary
    match_dict_all = read_all_raw_match_data()

    #create one dataframe with home and away player combined, per match and store in this dictionary
    aggregated_match_dict = {}

    for match in match_dict_all.keys():
        #create a full game timestamp
        match_dict_all[match] = create_fullgame_timestamp_variable(df=match_dict_all[match])

        match_dict_all[match] = reset_player_id(df=match_dict_all[match],full_match_id=match)

        #perform some initial data cleaning on data set
        match_dict_all[match] = initial_data_cleaning(df=match_dict_all[match])

        #obtain the match_id for the full match description
        match_id = match.replace("home_","").replace("away_","")

        #aggregate home and away player data in one dataframe
        if match_id in list(aggregated_match_dict.keys()):
            aggregated_match_dict[match_id] = combine_home_with_away_data(current_match_df=aggregated_match_dict[match_id],
                                                                        new_match_df=match_dict_all[match])
            
            #perform some data validations after home and away data is combined
            raw_data_validation(aggregated_match_dict[match_id])

            if match_id=="match4":
                train_or_test="test"
            else:
                train_or_test="train"
            aggregated_match_dict[match_id] = feature_engineering.create_new_features(aggregated_match_dict[match_id],train_or_test)

        else:
            aggregated_match_dict[match_id] = match_dict_all[match].copy()
    
    return aggregated_match_dict

def get_xlsx_files(folder_path):
    return [f for f in os.listdir(folder_path) if f.endswith(".xlsx")]

def get_csv_files(folder_path):
    return [f for f in os.listdir(folder_path) if f.endswith(".csv")]

def save_torch_model(model, file_name):
    #example:
    #save_model(model, "trained_model.pth")  # Save the model's state_dict

    torch.save(model.state_dict(), file_name)

# Load the model's state_dict
def load_torch_model(model, file_name):
    #example:
    # loaded_model = trackbox_LSTM(input_dim=10, hidden_dim=50, output_dim=2, num_layers=2, dropout=0.2)
    # loaded_model = load_model(loaded_model, "trained_model.pth")

    model.load_state_dict(torch.load(file_name))
    model.eval()  # Set the model to evaluation mode
    return model

def split_train_test_data(match_dict,match_ids):

    full_df = pd.DataFrame()
    for match_id in match_ids:
        assert len(match_dict[match_id])>0, f"dataframe for match_id '{match_id}' is empty in dictionary"

        if len(full_df)==0:
            full_df = match_dict[match_id].copy()
        else:
            full_df = pd.concat([full_df,match_dict[match_id]])
    return full_df.reset_index(drop=True)

def read_all_raw_match_data():

    folder_path = os.getcwd() + r"\data"

    #collect all game data in dictionary
    match_dict_all = {}

    match_xlsx_files = get_xlsx_files(folder_path)

    # for match_data in match_xlsx_files:

    #     match_name = match_data.replace(".xlsx","").lower()
    #     match_dict_all[match_name] = pd.read_excel(rf"data\{match_data}")

    match_csv_files = get_csv_files(folder_path)

    for match_data in match_csv_files:
        match_name = match_data.replace(".csv","").lower()
        match_dict_all[match_name] = pd.read_csv(rf"data\{match_data}",sep=",")

    return match_dict_all

def create_fullgame_timestamp_variable(df):

    max_timestamp_first_half = df[df["IdPeriod"]==1]["Time"].max()
    df = df.sort_values(by=["IdPeriod","Time"]).reset_index(drop=True)

    df["game_timestamp"] = np.where(
        df["IdPeriod"]==1,
        df["Time"],
        df["Time"]+max_timestamp_first_half+10
    )
    #need to add the timestamps of the first half to the second half
    #add an additional 1/10th of second, because first timestamp of second half is at 0

    assert len(df.sort_values("game_timestamp")[(df["game_timestamp"]-df["game_timestamp"].shift(1).fillna(-10))!=10])==0, "game_timestamp between two timestamps is bigger than 0.1 second."

    return df

def combine_home_with_away_data(current_match_df,new_match_df):

    if "ball_x" in new_match_df.columns and "ball_y" in new_match_df.columns:
        new_match_df = new_match_df.drop(["ball_x","ball_y"],axis=1)
    else:
        current_match_df = current_match_df[current_match_df["game_timestamp"]<=min(current_match_df["game_timestamp"].max(),new_match_df["game_timestamp"].max())].reset_index(drop=True)
        new_match_df = new_match_df[new_match_df["game_timestamp"]<=min(current_match_df["game_timestamp"].max(),new_match_df["game_timestamp"].max())].reset_index(drop=True)

    assert current_match_df["game_timestamp"].max()==new_match_df["game_timestamp"].max(), f"timestamp dont match between home and away"

    #remove duplicate column in new_match_df, before the merge
    new_match_df = new_match_df.drop(["game_timestamp"],axis=1)

    full_match_data = pd.merge(current_match_df,
                                  new_match_df,
                                  on=["MatchId","IdPeriod","Time"],
                                  how="inner")
    
    return full_match_data

def initial_data_cleaning(df):

    #in case of the test data not having the x and y coordinates of the ball, skip this cleaning step
    if "ball_x" in df.columns and "ball_y" in df.columns:

        #assuming NAs only occur at the end of a halftime: should check better
        df = df.dropna(subset=['ball_x', 'ball_y'])

        # max_timestamp_ball_x = df.dropna(subset=["ball_x"])["game_timestamp"].max()
        # max_timestamp_ball_y = df.dropna(subset=["ball_y"])["game_timestamp"].max()

        # #the maximum timestamp is based on ball_x and ball_y maximum coordinate

        # assert max_timestamp_ball_x==max_timestamp_ball_y

        # #filter dataframe to have the maximum game_timestamp capped at the last available x,y coordinate of the ball
        # df = df[df["game_timestamp"]<=min(max_timestamp_ball_x,max_timestamp_ball_y)].reset_index(drop=True)

    df["MatchId"] = df["MatchId"].str[-1].astype(int)
    
    return df

def plot_ball_trajectory(df):

    # Create a plot
    plt.figure(figsize=(8, 6))

    # Plot trajectories for each period separately
    for period in df['IdPeriod'].unique():
        subset = df[df['IdPeriod'] == period]
        plt.plot(subset['ball_x_pred'], subset['ball_y_pred'], marker='o', linestyle='-', label=f'Period {period}')

    # Set field boundaries
    plt.xlim(-5250, 5250)
    plt.ylim(-3400, 3400)

    # Labels and title
    plt.xlabel('Ball X Coordinate')
    plt.ylabel('Ball Y Coordinate')
    plt.title('Ball predicted Trajectory')
    plt.legend()
    # plt.grid(True)
    plt.show()

def plot_all_players_in_timestamp(id_half,time,match_df):

    #filter for timestamp within the game
    match_df = match_df[((match_df["IdPeriod"]==id_half)&(match_df["Time"]==time))].reset_index(drop=True)

    assert len(match_df)!=0, f"for specified timestamp, no player coordinate data is found."

    pattern = r"^(home|away)_[0-9]{1,2}_[xy]$"
    if "ball_x" in match_df.columns and "ball_y" in match_df.columns:
        match_df = match_df[["game_timestamp"]+[col for col in match_df.columns if re.match(pattern, col)]+["ball_x","ball_y"]]
    else:
        match_df = match_df[["game_timestamp"]+[col for col in match_df.columns if re.match(pattern, col)]]

    #delete any players with NA x and y coordinates
    match_df = match_df.dropna(axis=1)

    active_player_columns = [col for col in match_df.columns if re.match(pattern, col)]

    assert len(active_player_columns)==(11*2*2), f"in active_player_columns should always be 44 columns (11 players per team with a x and y coordinate column)"

    match_df_melted = match_df.melt(id_vars=["game_timestamp"],var_name="player_ball",value_name="coordinate")

    # Extract player name and coordinate type (x or y)
    match_df_melted[["player_name", "coordinate_type"]] = match_df_melted["player_ball"].str.rsplit("_", n=1, expand=True)
    match_df_melted["home_away_ball"] = np.where(
        match_df_melted["player_name"].str.contains("home_"),
        "home",
        np.where(
            match_df_melted["player_name"].str.contains("away_"),
            "away",
            "ball"
        )
    )

    # Pivot to separate x and y into columns
    df_pivoted = match_df_melted.pivot(index=["game_timestamp","player_name","home_away_ball"], columns="coordinate_type", values="coordinate").reset_index()

    # Rename columns
    df_pivoted = df_pivoted.rename(columns={"x": "x_coordinate", "y": "y_coordinate"})

    colors = {"home": "blue", "away": "red", "ball": "black"}

    for category, group in df_pivoted.groupby("home_away_ball"):
        plt.scatter(group["x_coordinate"], group["y_coordinate"], label=category, color=colors[category])

    # Set field boundaries
    plt.xlim(-5250, 5250)
    plt.ylim(-3400, 3400)

    # Add labels and legend
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.title('Player and Ball Positions')

    # Show plot
    plt.show()

def raw_data_validation(df):
    lol=""

    #count the number of active players per home and away team for every game_timestamp
    # df["home_player_count"] = df.filter(like="home_").notna().sum(axis=1)
    # df["away_player_count"] = df.filter(like="away_").notna().sum(axis=1)

    #TODO: bedenk een fix hiervoor
    # assert len(df[(df["home_player_count"]/2)!=11])==0, f"there should be 11 active home players per game_timestamp"
    # assert len(df[(df["away_player_count"]/2)!=11])==0, f"there should be 11 active away players per game_timestamp"
    # assert len(df[((df["home_player_count"]/2)+(df["away_player_count"]/2))!=22])==0, f"there should be 22 active home/away players per game_timestamp"

    # df = df.drop(["home_player_count","away_player_count"],axis=1)

    #TODO: check that each game_timestamp that has an x_coordinate for a player, for this player also a y coordinate exists

def sort_player_based_on_position(df,ascending_x,ascending_y,home_or_away):

    #collect all the player_ids with NA columns in the first game_timestamp
    na_columns = df.columns[df.iloc[0].isna()].tolist()
    na_columns_player_ids = [col.replace("_x","").replace("_y","").replace("away_","").replace("home_","") for col in na_columns]
    na_columns_player_ids = list(set(na_columns_player_ids))

    # extract the player_ids over which to sort for. Remove the player_ids which have NA value in the first game_timestamp
    player_ids = sorted(set(col.split("_")[1] for col in df.columns if (f"{home_or_away}_" in col)))
    player_ids = [player_id for player_id in player_ids if player_id not in na_columns_player_ids]

    # sort the player ids based on their x and y coordinate
    # also depending on how ascending_x and ascending_y are defined
    sorted_player_ids = sorted(
        player_ids, 
        key=lambda pid: (
            (df[f"{home_or_away}_{pid}_x"].iloc[0] if pd.notna(df[f"{home_or_away}_{pid}_x"].iloc[0]) else float('inf')) * (1 if ascending_x else -1),
            (df[f"{home_or_away}_{pid}_y"].iloc[0] if pd.notna(df[f"{home_or_away}_{pid}_y"].iloc[0]) else float('inf')) * (1 if ascending_y else -1)
        )
    )

    # Create new column order and place the NA valued player_ids at the back
    sorted_home_cols = [col for pid in sorted_player_ids for col in [f"{home_or_away}_{pid}_x", f"{home_or_away}_{pid}_y"]]
    na_sorted_home_cols = [col for pid in na_columns_player_ids for col in [f"{home_or_away}_{pid}_x", f"{home_or_away}_{pid}_y"]]

    #if ball coordinates are in the data, add them at the back of the dataframe
    if "ball_x" in df.columns and "ball_y" in df.columns:
        new_col_order = ["MatchId","IdPeriod","Time","game_timestamp"] + sorted_home_cols + na_sorted_home_cols + ["ball_x","ball_y"]
    else:
        new_col_order = ["MatchId","IdPeriod","Time","game_timestamp"] + sorted_home_cols + na_sorted_home_cols

    # Reorder DataFrame
    df = df[new_col_order]

    total_ordered_player_ids = sorted_player_ids + na_columns_player_ids

    return df,total_ordered_player_ids

# Function to rename column names
def rename_column(col,ordered_player_ids,ordered_player_id_to_index):
    for pid in ordered_player_ids:
        if f"_{pid}_" in col:  # Check if column contains the player ID
            return col.replace(f"_{pid}_", f"_{ordered_player_id_to_index[pid]}_")
    return col  # Return unchanged if no match

def reset_player_id(df,full_match_id):
    
    #obtain the average x coordinate of all home and away players on the first timestamp, respectively
    #this way it is determined whether home team starts on right or left side
    if "home_" in full_match_id.lower():
        average_x_coordinate_team = float(df.filter(like="home_").filter(like="_x").iloc[0].dropna().mean())
        home_or_away="home"
    else:
        average_x_coordinate_team = float(df.filter(like="away_").filter(like="_x").iloc[0].dropna().mean())
        home_or_away="away"

    #if average x coordinate of the team is negative, it means the team starts on the left side
    #therefore i want to sort the players based on ascending x coordinate and ascending y coordinate
    #if average x coordinate of the team is negative, i want sorting in descending order
    if average_x_coordinate_team<0:
        ascending_x = True
        ascending_y = True
    else:
        ascending_x=False
        ascending_y=False
    
    len_df_columns = len(df.columns)
    df,ordered_player_ids = sort_player_based_on_position(df,ascending_x=ascending_x,ascending_y=ascending_y,home_or_away=home_or_away)
    assert len(df.columns)==len_df_columns, f"sorting functions has either added or removed columns in the dataframe"

    ordered_player_id_to_index = {pid: str(idx) for idx, pid in enumerate(ordered_player_ids)}

    # Apply renaming to all columns
    df.columns = [rename_column(col,ordered_player_ids,ordered_player_id_to_index) for col in df.columns]

    return df