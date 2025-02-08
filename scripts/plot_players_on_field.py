from trackbox_case import load_data
import pickle

# aggregated_match_dict = load_data.load_match_data()
# Save to a pickle file
# with open("aggregated_match_dict_v2.pkl", "wb") as f:
#     pickle.dump(aggregated_match_dict, f)
# Load dictionary from a pickle file
with open("aggregated_match_dict_v2.pkl", "rb") as f:
    aggregated_match_dict = pickle.load(f)
print(f"match data loaded")

load_data.plot_all_players_in_timestamp(id_half=1,time=10000,match_df=aggregated_match_dict["match0"])