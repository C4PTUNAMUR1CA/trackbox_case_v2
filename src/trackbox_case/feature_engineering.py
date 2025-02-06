import numpy as np

def create_new_features(df):

    player_ids = [col.replace("_x","") for col in df.columns if ("home_" in col or "away_" in col)]
    player_ids = list(set(player_ids))

    df = create_speed_and_distance_covered_metric(df,player_ids)

    df = create_acceleration_metric(df,player_ids)

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
