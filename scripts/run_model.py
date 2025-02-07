from trackbox_case import load_data
from trackbox_case.obtain_model_forecast_and_accuracy import fit_model_and_obtain_forecast
import pickle

model_version="feature_list_2"
model_type = "LSTM"
hyperparameter_version="hyperparameters_1"

# aggregated_match_dict = load_data.load_match_data()
# Save to a pickle file
# with open("aggregated_match_dict_v2.pkl", "wb") as f:
#     pickle.dump(aggregated_match_dict, f)
# Load dictionary from a pickle file
with open("aggregated_match_dict_v2.pkl", "rb") as f:
    aggregated_match_dict = pickle.load(f)
print(f"match data loaded")

training_data = load_data.split_train_test_data(match_dict=aggregated_match_dict,match_ids=["match0","match1","match2","match3"])
print(f"training dataset created")

test_data = load_data.split_train_test_data(match_dict=aggregated_match_dict,match_ids=["match4"])
print(f"testing dataset created")

fit_model_and_obtain_forecast(model=model_type,
                              training_data=training_data,
                              testing_data=test_data,
                              model_version=model_version,
                              hyperparameter_version=hyperparameter_version)