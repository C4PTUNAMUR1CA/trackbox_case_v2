import re

def filling_missing_data(df):

    pattern = r"^(home|away)_[0-9]{1,2}_[xy]$"
    player_coordinate_cols = [col for col in df.columns if re.match(pattern, col)]

    # Assuming df is your DataFrame
    df[player_coordinate_cols] = df[player_coordinate_cols].apply(lambda x: x.fillna(0) if '_x' in x.name else x.fillna(-3400), axis=0)

    return df