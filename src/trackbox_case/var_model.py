import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from trackbox_case import config
from trackbox_case.utils import filling_missing_data

def deploy_var_model(training_df,testing_df,model_version):

    training_df = filling_missing_data(training_df)
    testing_df = filling_missing_data(testing_df)
    print(f"missing data for VAR model filled")

    # adfuller_test_for_stationarity(training_df)
    # adfuller_test_for_stationarity(testing_df)

    var_model_prediction(train_df=training_df,test_df=testing_df,model_version=model_version)

def adfuller_test_for_stationarity(df):

    df_adf_test = df.copy().drop(["MatchId"],axis=1)
    df_adf_test = df_adf_test[["IdPeriod","Time","game_timestamp","away_0_x","away_0_y",
                               "away_1_x","away_1_y","away_2_x","away_2_y","away_13_x","away_13_y",
                               "home_0_x","home_0_y",
                               "home_1_x","home_1_y","home_2_x","home_2_y","home_13_x","home_13_y"]]

    non_stationary_variables = []
    for col in df_adf_test.columns:
        print(f"adf test for column {col}")
        adf_result = adfuller(df_adf_test[col])

        #non-stationary explanatory variables cause the model to be unreliable or inaccurate
        #in result, extract the p-value for the specified variable
        #if Test statistic is large, it suggests the series is stationary
        # if p-value is below 5%, reject null hypothesis, meaning it is stationarity
        if adf_result[1]>=0.05:
            non_stationary_variables.append(col)
    if len(non_stationary_variables)==0:
        print(f"all variables are stationary, can continue with VAR model.")
    else:
        print(f"all variables are stationary, except {non_stationary_variables}")

def var_model_prediction(train_df,test_df,model_version):
    
    feature_columns = config.feature_columns[model_version]
    exogenous_columns = ["MatchId","IdPeriod","Time","game_timestamp"]
    feature_columns = [col for col in feature_columns if col not in exogenous_columns]
    to_be_forecast_columns = ["ball_x","ball_y"]

    endogenous_train_df = train_df[feature_columns+to_be_forecast_columns]

    test_df["ball_x"] = np.where(
        test_df["game_timestamp"]==0,
        0,
        None
    ).astype(float)
    test_df["ball_y"] = np.where(
        test_df["game_timestamp"]==0,
        0,
        None
    ).astype(float)
    endogenous_test_df = test_df[feature_columns+to_be_forecast_columns].reset_index(drop=True)

    exogenous_train_df = train_df[exogenous_columns]
    exogenous_test_df = test_df[exogenous_columns].reset_index(drop=True)

    model = VAR(endog=endogenous_train_df,
                # exog=exogenous_train_df
                )
    print(f"VAR model deployed with train_data")
    # lag_order = model.select_order(maxlags=5).aic
    # print(f"Optimal number of lags: {lag_order}")

    # 4. Fit the VAR model with the selected lag order
    model_fitted = model.fit(maxlags=1)
    print(f"VAR model fitted on training data")

    n_forecast = len(endogenous_train_df)

    #fit the model with forecast steps of one timestamp ahead
    forecast = model_fitted.forecast(endogenous_train_df.values,
                                     n_forecast,
                                    #  exogenous_test_df.values
                                     )

    lol=""

# Convert the forecasted values to a DataFrame for easier interpretation
# forecast_df = pd.DataFrame(forecast, columns=['home_200_x', 'home_200_y', 'away_100_x', 'away_100_y', 'ball_x', 'ball_y'])
# print(f"Forecasted values: \n{forecast_df}")

# # 6. Plot the original and predicted ball_x, ball_y coordinates (just an example)
# plt.plot(df['timestamp'], df['ball_x'], label='Actual Ball X')
# plt.plot(df['timestamp'][-forecast_steps:], forecast_df['ball_x'], label='Predicted Ball X', linestyle='--')
# plt.legend()
# plt.title('Ball X Coordinate Prediction')
# plt.show()