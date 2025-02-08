import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from trackbox_case import config
from trackbox_case.utils import filling_missing_data
import pandas as pd
from trackbox_case.load_data import save_torch_model
from datetime import datetime
from trackbox_case.feature_engineering import create_ball_related_features

# Define LSTM Model
class trackbox_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(trackbox_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, int(hidden_dim), int(num_layers), batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # Get LSTM output
        out = self.fc(lstm_out)  # out and push it through the fully connected layer to get a regression result: Y = w*H + b

        # Apply sigmoid to the last output neuron for binary classification (possession)
        out[:,-1] = torch.sigmoid(out[:,-1])  # Apply only to the last output variable (possession_home_boolean)

        return out

def perform_kfold_cross_validation(full_training_df,
                                   feature_col_list,
                                   predictor_col_list,
                                   hyperparameters_dict):

    #full training set contains match ids 0 to 3.
    match_id_list = [0,1,2,3]

    total_validation_rmse = 0
    #every cross validation, there is gonna be a validation on one of the matches
    for match_id in match_id_list:
        print(f"k-fold cross validation step 1, with validation on match_id {match_id}")
        #create training and validation dataframes for the explanatory variables
        x_validation_df = full_training_df[full_training_df[f"match_{match_id}_boolean"]==1][feature_col_list].reset_index(drop=True).values
        x_training_df = full_training_df[full_training_df[f"match_{match_id}_boolean"]==0][feature_col_list].reset_index(drop=True).values

        #create training and validation dataframes for the predictory variables
        y_validation_df = full_training_df[full_training_df[f"match_{match_id}_boolean"]==1][predictor_col_list].reset_index(drop=True).values
        y_training_df = full_training_df[full_training_df[f"match_{match_id}_boolean"]==0][predictor_col_list].reset_index(drop=True).values

        #translate pandas dataframe to a pytorch table format
        x_train_tensor = torch.tensor(x_training_df,dtype=torch.float32)
        y_train_tensor = torch.tensor(y_training_df,dtype=torch.float32)
        x_validation_tensor = torch.tensor(x_validation_df,dtype=torch.float32)
        y_validation_tensor = torch.tensor(y_validation_df,dtype=torch.float32)

        best_model, train_rmse, validation_rmse = train_model(x_train_games=x_train_tensor,
                                                            y_train_games=y_train_tensor,
                                                            x_validation_games=x_validation_tensor,
                                                            y_validation_games=y_validation_tensor,
                                                            input_dim=hyperparameters_dict["input_dim"],
                                                            hidden_dim=hyperparameters_dict["hidden_dim"],
                                                            output_dim=hyperparameters_dict["output_dim"],
                                                            num_layers=hyperparameters_dict["num_layers"],
                                                            epochs=hyperparameters_dict["epochs"],
                                                            batch_size=hyperparameters_dict["batch_size"],
                                                            learning_rate=hyperparameters_dict["learning_rate"],
                                                            validation_boolean=True)
        
        #accumulate the validation rmse of all kfold models
        total_validation_rmse += float(validation_rmse)

    #calculate average validation rmse
    average_validation_rmse = total_validation_rmse/len(match_id_list)

    return average_validation_rmse

def train_model(x_train_games,y_train_games,
                x_validation_games,y_validation_games, 
                input_dim, hidden_dim, output_dim, 
                num_layers, epochs, batch_size, learning_rate,
                validation_boolean):
    
    #combine explanatory and predictory variables in one tensordataset, for training and validation respectively
    train_dataset = TensorDataset(x_train_games, y_train_games)
    train_loader = DataLoader(train_dataset, batch_size=int(batch_size), shuffle=False)

    model = trackbox_LSTM(input_dim, hidden_dim, output_dim, num_layers, dropout=0.2)
    #rather than SGD, make use of Adam algorithm to update weights and biases
    # compared to SGD, Adam adapts the learning rate and thus is more efficient than SGD
    #theta = theta - ita * delta_J(theta) -> ita=learning_rate, theta=weight or bias, delta_J(theta)=gradient of loss function, w.r.t. theta
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #make use of the mean squared error loss function
    loss_function = nn.MSELoss()

    for epoch in range(epochs):
        if epoch % 10 ==0 or epoch==0:
            print(f"Epoch: {str(epoch)}")

        model.train()
        for x_batch,y_batch in train_loader:
            #calculate predicted y variables for specified batch of X
            y_pred = model(x_batch)
            #compute loss function, used to update the weights in backward propagation
            loss = loss_function(y_pred, y_batch)
            #reset the gradients of all model parameters, then compute new gradients with backward propagation
            #otherwise gradients accumulate by default in pytorch, leading to incorrect weight updates
            optimizer.zero_grad()
            loss.backward()
            #update the model's parameters (weight and bias) based on the gradients computer during backward propagation
            optimizer.step()
        #validate every so epochs with RMSE
        if epoch % 25 !=0:
            continue
        # Validation
        model.eval()
        #this section disables gradient tracking, to reduce memory usage and speed up inference
        with torch.no_grad():
            y_pred_training = model(x_train_games)
            train_rmse = np.sqrt(loss_function(y_pred_training, y_train_games))

            if validation_boolean:
                y_pred_validation = model(x_validation_games)
                validation_rmse = np.sqrt(loss_function(y_pred_validation,y_validation_games))

        if validation_boolean:
            print(f"Epoch {epoch+1}/{epochs}, Train RMSE: {train_rmse}, Validation RMSE: {validation_rmse}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, Train RMSE: {train_rmse}")

    if validation_boolean:
        return model, train_rmse, validation_rmse
    else:
        return model, train_rmse

def one_hot_encode_categorical_variables(df):

    match_id_list = [0,1,2,3,4]
    for match_id in match_id_list:
        df[f"match_{str(match_id)}_boolean"] = np.where(
            df["MatchId"]==match_id,
            1,
            0
        )
    df = df.drop(["MatchId"],axis=1)

    period_list = [1,2]
    for period in period_list:
        df[f"period_{str(period)}_boolean"] = np.where(
            df["IdPeriod"]==period,
            1,
            0
        )
    df = df.drop(["IdPeriod"],axis=1)

    return df

def standardise_data(df):
    
    cols_to_standardise = [col for col in df.columns if "_boolean" not in col]

    #standardised values = (x - miu)/sigma -> miu mean of the column and sigma is the standard deviation of column
    scaler = StandardScaler()

    #before standardisation, collect the mean and stdev of ball_x and ball_y (from training_data)
    scaler_predictors_info_dict = {}
    for col in cols_to_standardise:
        scaler_predictors_info_dict[col]={
            "mean":float(df[col].mean()),
            "stdev":float(df[col].std())
        }

    df[cols_to_standardise] = scaler.fit_transform(df[cols_to_standardise])

    return df,scaler_predictors_info_dict

def standardise_with_train_data(df,scaler_info):

    cols_to_standardise = [col for col in df.columns if "_boolean" not in col]

    for col in cols_to_standardise:
        df[col] = (df[col] - scaler_info[col]["mean"])/scaler_info[col]["stdev"]
    
    return df

def create_lstm_predictions(test_df,feature_col_list,predictor_col_list,lstm_model,scaler_info):

    #create X dataframe from the testing set and make it into tensor data format
    # x_test_df = test_df[feature_col_list].reset_index(drop=True).values
    # x_test_tensor = torch.tensor(x_test_df,dtype=torch.float32)

    #introduce the predictor columns to test_df
    test_df[predictor_col_list] = [0,0,0]

    #collect all predicted ball_x and ball_y values in here
    predicted_positions = []
    lstm_model.eval()

    for i in range(len(test_df)):
        if i % 1000==0:
            print(f"it is the {str(i)}th forecast out of {str(len(test_df))}")
        if test_df.loc[i,"Time"]==0:
            #at the start of each half, predict that the ball is at 0
            ball_x_pred = 0
            ball_y_pred = 0
            #TODO: fix this prediction
            possession_home_boolean_pred=1

            #update ball related features at current row
            test_df.loc[i,predictor_col_list] = [ball_x_pred,ball_y_pred,possession_home_boolean_pred]

            #store predicted coordinates
            predicted_positions.append([ball_x_pred,ball_y_pred,possession_home_boolean_pred])
        else:
            test_df_to_ith_sample = test_df.copy()[0:(i+1)]
            #update features, such that at time t the ball-related features are created with the forecast ball coordinates at time t-1
            test_df_to_ith_sample = create_ball_related_features(test_df_to_ith_sample,train_or_test="train")

            test_df_to_ith_sample = standardise_with_train_data(test_df_to_ith_sample,scaler_info)

            x_test_df = test_df_to_ith_sample[feature_col_list].reset_index(drop=True).values
            x_test_tensor = torch.tensor(x_test_df,dtype=torch.float32)

            input_features = x_test_tensor[i].clone()

            # Reshape for LSTM: (batch=1, sequence=1, features)
            input_features = input_features.unsqueeze(0).unsqueeze(0)

            #obtain predictions for the testing set
            #torch.no_grad(): states explicitly to pytorch that no gradient is being calculated here
            with torch.no_grad():
                y_pred_test = lstm_model(input_features)

            # Get predictions
            ball_x_pred_standardised, ball_y_pred_standardised, possession_home_boolean_pred_standardised = y_pred_test.squeeze().tolist()

            ball_x_pred = ball_x_pred_standardised*scaler_info["ball_x"]["stdev"] + scaler_info["ball_x"]["mean"]
            ball_y_pred = ball_y_pred_standardised*scaler_info["ball_y"]["stdev"] + scaler_info["ball_y"]["mean"]
            #sigmoid valued predicted value, so if above 0.5, then convert boolean predicted variable to 1, otherwise 0
            if possession_home_boolean_pred_standardised>0.5:
                possession_home_boolean_pred = 1
            else:
                possession_home_boolean_pred = 0

            # Ensure predictions stay within field boundaries
            ball_x_pred = float(np.clip(ball_x_pred, -config.length_field_in_metres*100/2, config.length_field_in_metres*100/2))
            ball_y_pred = float(np.clip(ball_y_pred, -config.width_field_in_metres*100/2, config.width_field_in_metres*100/2))

            #update ball related features at current row
            test_df.loc[i,predictor_col_list] = [ball_x_pred,ball_y_pred,possession_home_boolean_pred]

            #store predicted coordinates
            predicted_positions.append([ball_x_pred,ball_y_pred,possession_home_boolean_pred])

    df_testing_predictions = pd.DataFrame(predicted_positions, columns=[col + "_pred" for col in predictor_col_list])

    return df_testing_predictions

def deploy_lstm_model(training_df,test_df,model_version,hyperparameters_dict,kfold_cross_validation_boolean=False):

    training_df = filling_missing_data(training_df)
    test_df = filling_missing_data(test_df)

    #chosen feature columns for lstm
    chosen_features = config.feature_columns[model_version]

    predictor_col_list = ["ball_x","ball_y","possession_home_boolean"]

    #extract only relevant feature columns and predictor variables
    training_df = training_df[chosen_features+predictor_col_list]
    test_df = test_df[chosen_features]

    #one-hot encode the categorical variables into dummy variables
    training_df = one_hot_encode_categorical_variables(df=training_df)
    test_df = one_hot_encode_categorical_variables(df=test_df)

    #standardise the data such that no feature weighs heavier within the training data
    #also helps training the LSTM more efficiently
    training_df,scaler_predictors_training_info = standardise_data(df=training_df)

    #create feature column list for predicting ball_x and ball_y
    feature_col_list = [col for col in training_df if col not in predictor_col_list]

    #define what the input_dimension should be for the lstm model
    hyperparameters_dict["input_dim"] = len(feature_col_list)
    hyperparameters_dict["output_dim"] = len(predictor_col_list)

    #form the x and y tables for training the model
    training_df = training_df[feature_col_list+predictor_col_list]

    if kfold_cross_validation_boolean:
        average_validation_rmse = perform_kfold_cross_validation(training_df,feature_col_list,predictor_col_list,hyperparameters_dict)
        
        return average_validation_rmse
    else:

        #create training dataframes for the explanatory variables
        x_training_df = training_df[feature_col_list].reset_index(drop=True).values

        #create training dataframes for the predictory variables
        y_training_df = training_df[predictor_col_list].reset_index(drop=True).values

        #translate pandas dataframe to a pytorch table format
        x_train_tensor = torch.tensor(x_training_df,dtype=torch.float32)
        y_train_tensor = torch.tensor(y_training_df,dtype=torch.float32)

        best_model, train_rmse = train_model(x_train_games=x_train_tensor,
                                                            y_train_games=y_train_tensor,
                                                            x_validation_games=None,
                                                            y_validation_games=None,
                                                            input_dim=hyperparameters_dict["input_dim"],
                                                            hidden_dim=hyperparameters_dict["hidden_dim"],
                                                            output_dim=hyperparameters_dict["output_dim"],
                                                            num_layers=hyperparameters_dict["num_layers"],
                                                            epochs=hyperparameters_dict["epochs"],
                                                            batch_size=hyperparameters_dict["batch_size"],
                                                            learning_rate=hyperparameters_dict["learning_rate"],
                                                            validation_boolean=False)
        
        #store the optimised model
        model_stored_name = f"trained_models/lstm_model_{datetime.today().strftime('%Y%m%d%H%M')}.pth"
        save_torch_model(best_model,model_stored_name)

        df_testing_predictions = create_lstm_predictions(test_df=test_df,
                                                            feature_col_list=feature_col_list,
                                                            predictor_col_list=predictor_col_list,
                                                            lstm_model=best_model,
                                                            scaler_info=scaler_predictors_training_info)
        
        #stored_predictions
        prediction_table_name = f"prediction_output/lstm_prediction_{datetime.today().strftime('%Y%m%d%H%M')}.pkl"
        df_testing_predictions.to_pickle(prediction_table_name)

        return df_testing_predictions, train_rmse, model_stored_name, prediction_table_name
        
        #create X dataframe from the testing set and make it into tensor data format
        # x_test_df = test_df[feature_col_list].reset_index(drop=True).values
        # x_test_tensor = torch.tensor(x_test_df,dtype=torch.float32)

        # #obtain predictions for the testing set
        # y_pred_test = best_model(x_test_tensor)

        # #transform y_pred_test tensor table to numpy table
        # y_pred_numpy  = y_pred_test.detach().cpu().numpy()

        # #translate numpy table into dataframe
        # df_testing_predictions = pd.DataFrame(y_pred_numpy, columns=["ball_x_pred", "ball_y_pred"])
        
        # #destandardise data, using mean and stdev from training_data, since this information is not available for the testing set
        # df_testing_predictions["ball_x_pred"] = df_testing_predictions["ball_x_pred"]*scaler_predictors_training_info["ball_x"]["stdev"] + scaler_predictors_training_info["ball_x"]["mean"]
        # df_testing_predictions["ball_y_pred"] = df_testing_predictions["ball_y_pred"]*scaler_predictors_training_info["ball_y"]["stdev"] + scaler_predictors_training_info["ball_y"]["mean"]
        
        # #if some predicted results are out of bounds of x and y coordinates (field is 105m long by 68m wide), then set them within the boundaries
        # df_testing_predictions["ball_x_pred"] = np.where(
        #     df_testing_predictions["ball_x_pred"]<(-1*round(config.length_field_in_metres*100/2,0)),
        #     (-1*round(config.length_field_in_metres*100/2,0)),
        #     np.where(
        #         df_testing_predictions["ball_x_pred"]>round(config.length_field_in_metres*100/2,0),
        #         round(config.length_field_in_metres*100/2,0),
        #         df_testing_predictions["ball_x_pred"]
        #     )
        # )
        # df_testing_predictions["ball_y_pred"] = np.where(
        #     df_testing_predictions["ball_y_pred"]<(-1*round(config.width_field_in_metres*100/2,0)),
        #     (-1*round(config.width_field_in_metres*100/2,0)),
        #     np.where(
        #         df_testing_predictions["ball_y_pred"]>round(config.width_field_in_metres*100/2,0),
        #         round(config.width_field_in_metres*100/2,0),
        #         df_testing_predictions["ball_y_pred"]
        #     )
        # )

        # #stored_predictions
        # prediction_table_name = f"prediction_output/lstm_prediction_{datetime.today().strftime('%Y%m%d%H%M')}.pkl"
        # df_testing_predictions.to_pickle(prediction_table_name)

        # return df_testing_predictions, train_rmse, model_stored_name, prediction_table_name
