import pandas as pd

# Example DataFrame
data = {
    "game_timestamp": [0],
    "home_200_x": [-100], "home_200_y": [20],
    "home_100_x": [-100], "home_100_y": [-10],
    "home_50_x": [None], "home_50_y": [None],
    "home_70_x": [None], "home_70_y": [None]
}
df = pd.DataFrame(data)

ascending_x = False
ascending_y = False

na_columns = df.columns[df.iloc[0].isna()].tolist()
na_columns_player_ids = [col.replace("_x","").replace("_y","").replace("away_","").replace("home_","") for col in na_columns]
na_columns_player_ids = list(set(na_columns_player_ids))

# Extract unique player IDs from home_* columns
player_ids = sorted(set(col.split("_")[1] for col in df.columns if "home_" in col))
player_ids = [player_id for player_id in player_ids if player_id not in na_columns_player_ids]

# Sort player IDs based on first row values, placing NaNs at the end
sorted_player_ids = sorted(
    player_ids, 
    key=lambda pid: (
        (df[f"home_{pid}_x"].iloc[0] if pd.notna(df[f"home_{pid}_x"].iloc[0]) else float('inf')) * (1 if ascending_x else -1),
        (df[f"home_{pid}_y"].iloc[0] if pd.notna(df[f"home_{pid}_y"].iloc[0]) else float('inf')) * (1 if ascending_y else -1)
    )
)

# Create new column order
sorted_home_cols = [col for pid in sorted_player_ids for col in [f"home_{pid}_x", f"home_{pid}_y"]]
na_sorted_home_cols = [col for pid in na_columns_player_ids for col in [f"home_{pid}_x", f"home_{pid}_y"]]
new_col_order = ["game_timestamp"] + sorted_home_cols + na_sorted_home_cols

# Reorder DataFrame
df = df[new_col_order]