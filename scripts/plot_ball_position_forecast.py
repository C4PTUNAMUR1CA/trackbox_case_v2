from trackbox_case import load_data
import pickle
import pandas as pd

#prediction lstm with feature_list_2, full training: 
# ball_prediction_pickle_file = f"prediction_output/lstm_prediction_202502071032.pkl"

train_or_prediction_data = "prediction"
training_match_id = "match0"
if train_or_prediction_data=="prediction":
    ball_prediction_pickle_file = f"prediction_output/lstm_prediction_202502071032.pkl"

# aggregated_match_dict = load_data.load_match_data()
# Save to a pickle file
# with open("aggregated_match_dict_v2.pkl", "wb") as f:
#     pickle.dump(aggregated_match_dict, f)
# Load dictionary from a pickle file
with open("aggregated_match_dict_v2.pkl", "rb") as f:
    aggregated_match_dict = pickle.load(f)
print(f"match data loaded")

if train_or_prediction_data=="prediction":
    test_data = load_data.split_train_test_data(match_dict=aggregated_match_dict,match_ids=["match4"])
    print(f"testing dataset created")

    prediction_df = pd.read_pickle(ball_prediction_pickle_file)

    full_df = pd.concat([test_data,prediction_df],axis=1)
elif train_or_prediction_data=="train":
    #collect the train_data for specified match_id
    train_data = load_data.split_train_test_data(match_dict=aggregated_match_dict,match_ids=[training_match_id])
    print(f"testing dataset created")

    full_df = train_data.rename(columns={"ball_x":"ball_x_pred","ball_y":"ball_y_pred"})

plot_ball_trajectory_df = full_df.copy()[["IdPeriod","Time","ball_x_pred","ball_y_pred"]]
load_data.plot_ball_trajectory(full_df)

full_df = full_df.rename(columns={"ball_x_pred":"ball_x","ball_y_pred":"ball_y"})

load_data.plot_all_players_in_timestamp(id_half=1,time=10000,match_df=full_df)