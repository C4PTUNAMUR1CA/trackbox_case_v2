from trackbox_case import var_model
from trackbox_case import lstm_model
from trackbox_case import config
from datetime import datetime
import pandas as pd

def update_model_output_table(model,train_rmse,model_stored_name,prediction_table_name,model_version,hyperparameter_version):

    timestamp = datetime.now().strftime("%Y%m%d%H%M")

    new_row = {
        "model_name": model,
        "model_stored_name": model_stored_name,
        "prediction_table_name":prediction_table_name,
        "train_rmse": train_rmse,
        "model_version": model_version,
        "hyperparameter_version": hyperparameter_version,
        "timestamp": timestamp
    }

    df_output_data = pd.read_pickle('model_output_data.pkl')

    df_output_data = pd.concat([df_output_data, pd.DataFrame([new_row])], ignore_index=True)
    df_output_data = df_output_data.reset_index(drop=True)

    df_output_data.to_pickle("model_output_data.pkl")
    print("model_output_data.pkl has been updated")

def fit_model_and_obtain_forecast(model,training_data,testing_data,model_version,hyperparameter_version={}):

    if model=="VAR":
        var_model.deploy_var_model(training_df=training_data,
                                   testing_df=testing_data,
                                   model_version=model_version)
    elif model=="LSTM":
        hyperparameters = config.hyperparameter_lstm[hyperparameter_version]

        df_predictions, train_rmse, model_stored_name, prediction_table_name = lstm_model.deploy_lstm_model(training_df=training_data,
                                                                                                            test_df=testing_data,
                                                                                                            model_version=model_version,
                                                                                                            hyperparameters_dict=hyperparameters)
        
    update_model_output_table(model=model,
                              train_rmse=train_rmse,
                              model_stored_name=model_stored_name,
                              prediction_table_name=prediction_table_name,
                              model_version=model_version,
                              hyperparameter_version=hyperparameter_version)
    

        

        
    

