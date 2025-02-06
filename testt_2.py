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

# Example ordered player IDs
ordered_player_ids = ["100", "200", "50"]  # This is the ordered list

# Create a mapping from player ID to its index in the ordered list
player_id_to_index = {pid: str(idx) for idx, pid in enumerate(ordered_player_ids)}

# Function to rename column names
def rename_column(col):
    for pid in ordered_player_ids:
        if f"_{pid}_" in col:  # Check if column contains the player ID
            return col.replace(f"_{pid}_", f"_{player_id_to_index[pid]}_")
    return col  # Return unchanged if no match

# Apply renaming to all columns
df.columns = [rename_column(col) for col in df.columns]

print(df.columns)  # Check new column names