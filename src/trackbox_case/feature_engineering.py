import numpy as np
import re

def create_new_features(df,train_or_test="train"):

    player_ids = [col.replace("_x","").replace("_y","") for col in df.columns if ("home_" in col or "away_" in col)]
    player_ids = list(set(player_ids))

    df = create_speed_and_distance_covered_metric(df,player_ids)

    df = create_acceleration_metric(df,player_ids)

    df = create_ball_related_features(df,train_or_test)

    return df

def create_ball_related_features(df,train_or_test="train"):

    pattern = r"^(home|away)_[0-9]{1,2}_[xy]$"

    player_ids = [col.replace("_x","").replace("_y","") for col in df.columns if re.match(pattern, col)]
    player_ids = list(set(player_ids))

    df = create_distance_from_ball_metric(df,player_ids,train_or_test)

    return df

def create_speed_and_distance_covered_metric(df,player_ids):

    df["delta_time"] = 0.1
    for player_id in player_ids:

        #calculate delta distances between x and y coordinates at time t and t-1.
        #divide by 100 to convert distance from cm to m
        df[f"delta_distance_{player_id}_x"] = df[f"{player_id}_x"] - df[f"{player_id}_x"].shift(1)
        df[f"delta_distance_{player_id}_x"] = df[f"delta_distance_{player_id}_x"]/100
        df[f"delta_distance_{player_id}_y"] = df[f"{player_id}_y"] - df[f"{player_id}_y"].shift(1)
        df[f"delta_distance_{player_id}_y"] = df[f"delta_distance_{player_id}_y"]/100

        #create variable that calculates the distance covered between two timestamps
        df[f"distance_covered_{player_id}"] = np.sqrt(df[f"delta_distance_{player_id}_x"]**2 + df[f"delta_distance_{player_id}_y"]**2)
        df[f"speed_{player_id}"] = df[f"distance_covered_{player_id}"]/df["delta_time"]
        #at the start of each half, set the speed and distance covered to zero to not create any unexplainable speeds between two halfs
        df[f"speed_{player_id}"] = df[f"speed_{player_id}"].fillna(0)
        df[f"speed_{player_id}"] = np.where(
            df["Time"]==0,
            0,
            df[f"speed_{player_id}"]
        )
        df[f"distance_covered_{player_id}"] = df[f"distance_covered_{player_id}"].fillna(0)
        df[f"distance_covered_{player_id}"] = np.where(
            df["Time"]==0,
            0,
            df[f"distance_covered_{player_id}"]
        )

        #TODO: what to do about this variable: not really stationary (as it only can grow)
        # df[f"distance_covered_since_start_{player_id}"] = df[f"distance_covered_{player_id}"].cumsum()

        df = df.drop([f"delta_distance_{player_id}_x",f"delta_distance_{player_id}_y"],axis=1)

    return df

def create_acceleration_metric(df,player_ids):
    
    df["delta_time"] = 0.1
    for player_id in player_ids:
        df[f"acceleration_{player_id}"] = (df[f"speed_{player_id}"] - df[f"speed_{player_id}"].shift(1))/df["delta_time"]
        df[f"acceleration_{player_id}"] = df[f"acceleration_{player_id}"].fillna(0)

        df[f"acceleration_{player_id}"] = np.where(
            df[f"Time"]==0,
            0,
            df[f"acceleration_{player_id}"]
        )
    df = df.drop(["delta_time"],axis=1)
    
    return df

def create_distance_from_ball_metric(df,player_ids,train_or_test):

    if train_or_test=="test":
        for player_id in player_ids:
            df[f"distance_from_ball_{player_id}"] = np.nan
        return df

    #since in testing set, the ball-related data at time t will be calculated with ball coordinates at time t-1
    #the same will be done for training data
    df["ball_x_shifted"] = df["ball_x"].shift(1).fillna(0)
    df["ball_x_shifted"] = np.where(
        df["Time"]==0,
        0,
        df["ball_x_shifted"]
    )
    df["ball_y_shifted"] = df["ball_y"].shift(1).fillna(0)
    df["ball_y_shifted"] = np.where(
        df["Time"]==0,
        0,
        df["ball_y_shifted"]
    )

    for player_id in player_ids:
        #obtain x and y coordinate deltas between player and ball
        df[f"x_delta_distance"] = (df["ball_x_shifted"] - df[f"{player_id}_x"]).abs()
        df[f"y_delta_distance"] = (df["ball_y_shifted"] - df[f"{player_id}_y"]).abs()

        #distance in metres from ball
        df[f"distance_from_ball_{player_id}"] = np.sqrt((df[f"x_delta_distance"]**2)+(df[f"y_delta_distance"]**2))/100
        #TODO: not sure if this is correct approach for fillna
        df[f"distance_from_ball_{player_id}"] = df[f"distance_from_ball_{player_id}"].fillna(0)
    df = df.drop(["x_delta_distance","y_delta_distance","ball_x_shifted","ball_y_shifted"],axis=1)

    return df

def create_ball_speed_metric(df,train_or_test):

    if train_or_test=="test":
        df["speed_ball"]=np.nan
        return df

    df["delta_time"] = 0.1
    df[f"delta_distance_ball_x"] = (df[f"ball_x"].shift(1) - df[f"ball_x"].shift(2))/100
    df[f"delta_distance_ball_y"] = (df[f"ball_y"].shift(1) - df[f"ball_y"].shift(2))/100
    df[f"distance_covered_ball"] = np.sqrt(df[f"delta_distance_ball_x"]**2 + df[f"delta_distance_ball_y"]**2)
    df[f"speed_ball"] = df[f"distance_covered_ball"]/df["delta_time"]
    df["speed_ball"] = np.where(
        df["Time"] in [0,10],
        0,
        df["speed_ball"]
    )
    df = df.drop(["delta_time","delta_distance_ball_x","delta_distance_ball_y","distance_covered_ball"],axis=1)

    return df

def assign_possession_to_team(df):

    #create column lists of all home and away players, separately
    home_players_cols = [col for col in df.columns if ("distance_from_ball_" in col and "_home_" in col)]
    away_players_cols = [col for col in df.columns if ("distance_from_ball_" in col and "_away_" in col)]

    #find for each timestamp the minimum distance of any home player from the ball, the same for away player
    df["home_min_distance_to_ball"] = df[home_players_cols].min(axis=1)
    df["away_min_distance_to_ball"] = df[away_players_cols].min(axis=1)
    
    #TODO how to do regarding standardisation and forecast, forecast will never be 0 or 1
    #create two boolean variables as to which team has possession
    df["possession_home_boolean"]= np.where(
        df["home_min_distance_to_ball"]<df["away_min_distance_to_ball"],
        1,
        0
    )
    df["possession_away_boolean"]= np.where(
        df["home_min_distance_to_ball"]<df["away_min_distance_to_ball"],
        0,
        1
    )

    return df
