import pickle
from trackbox_case import load_data
import itertools
import pandas as pd
from trackbox_case import lstm_model

model_version="feature_list_1"
model_type = "LSTM"
hyperparameter_version="hyperparameters_1"

# aggregated_match_dict = load_data.load_match_data()
# Save to a pickle file
# with open("aggregated_match_dict.pkl", "wb") as f:
#     pickle.dump(aggregated_match_dict, f)
# Load dictionary from a pickle file
with open("aggregated_match_dict.pkl", "rb") as f:
    aggregated_match_dict = pickle.load(f)
print(f"match data loaded")

training_data = load_data.split_train_test_data(match_dict=aggregated_match_dict,match_ids=["match0","match1","match2","match3"])
print(f"training dataset created")

test_data = load_data.split_train_test_data(match_dict=aggregated_match_dict,match_ids=["match4"])
print(f"testing dataset created")

#list the hyperparameters and possible values for a hyperparameter tuning grid search
hidden_dim_list = [50,100]
num_layers_list = [1,2]
epochs_list = [50,100]
batch_size_list = [32,64]
learning_rate_list = [0.001,0.0001]
# hidden_dim_list = [50,100,150]
# num_layers_list = [1,2,3]
# epochs_list = [50,100,150]
# batch_size_list = [32,64,128]
# learning_rate_list = [0.01,0.001,0.0001]

hyperparameter_list = ["hidden_dim", "num_layers", "epochs", "batch_size", "learning_rate"]

# Create a grid of all possible hyperparameter tuning combinations
hyperparameter_grid = list(itertools.product(hidden_dim_list, num_layers_list, epochs_list, batch_size_list, learning_rate_list))

# Create a DataFrame
df_hyperparameter_grid = pd.DataFrame(hyperparameter_grid, columns=hyperparameter_list)
df_hyperparameter_grid["average_validation_rmse"] = 0

#code for hyperparameter tuning grid search
for i in range(len(df_hyperparameter_grid)):
    print(f"grid search number {i} out of {len(df_hyperparameter_grid)}")
    hyperparameter_lstm = {}
    for hyperparameter in df_hyperparameter_grid:
        hyperparameter_lstm[hyperparameter] = df_hyperparameter_grid.loc[i,hyperparameter]

    average_validation_rmse = lstm_model.deploy_lstm_model(training_df=training_data,
                                                        test_df=test_data,
                                                        model_version=model_version,
                                                        hyperparameters_dict=hyperparameter_lstm,
                                                        kfold_cross_validation_boolean=True)
    df_hyperparameter_grid.loc[i,"average_validation_rmse"] = average_validation_rmse
    print(df_hyperparameter_grid)

print(f"""the best performing hyperparameter combination is: {df_hyperparameter_grid[df_hyperparameter_grid['average_validation_time']==
                                                                                   df_hyperparameter_grid['average_validation_time'].min()][[
                                                                                       'hidden_dim','num_layers','epochs','batch_size','learning_rate'
                                                                                   ]]}""")

df_hyperparameter_grid.to_pickle("lstm_hyperparameter_tuning.pkl")

#how to speed up this code?
#random search rather than grid search
#smaller training and validation datasets