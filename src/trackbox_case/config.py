feature_columns = {
    "feature_list_1":[
        "MatchId","IdPeriod","Time","game_timestamp",
        "home_0_x", "home_0_y", "home_1_x", "home_1_y", "home_2_x", "home_2_y", "home_3_x", "home_3_y", "home_4_x", "home_4_y", "home_5_x", "home_5_y", 
        "home_6_x", "home_6_y", "home_7_x", "home_7_y", "home_8_x", "home_8_y", "home_9_x", "home_9_y", "home_10_x", "home_10_y", "home_11_x", "home_11_y", 
        "home_12_x", "home_12_y", "home_13_x", "home_13_y",
        "away_0_x", "away_0_y", "away_1_x", "away_1_y", "away_2_x", "away_2_y", "away_3_x", "away_3_y", "away_4_x", "away_4_y", "away_5_x", "away_5_y", 
        "away_6_x", "away_6_y", "away_7_x", "away_7_y", "away_8_x", "away_8_y", "away_9_x", "away_9_y", "away_10_x", "away_10_y", "away_11_x", "away_11_y", 
        "away_12_x", "away_12_y", "away_13_x", "away_13_y",
    ],
    "feature_list_2":[
        "MatchId","IdPeriod","Time","game_timestamp",
        "home_0_x", "home_0_y", "home_1_x", "home_1_y", "home_2_x", "home_2_y", "home_3_x", "home_3_y", "home_4_x", "home_4_y", "home_5_x", "home_5_y", 
        "home_6_x", "home_6_y", "home_7_x", "home_7_y", "home_8_x", "home_8_y", "home_9_x", "home_9_y", "home_10_x", "home_10_y", "home_11_x", "home_11_y", 
        "home_12_x", "home_12_y", "home_13_x", "home_13_y",
        "away_0_x", "away_0_y", "away_1_x", "away_1_y", "away_2_x", "away_2_y", "away_3_x", "away_3_y", "away_4_x", "away_4_y", "away_5_x", "away_5_y", 
        "away_6_x", "away_6_y", "away_7_x", "away_7_y", "away_8_x", "away_8_y", "away_9_x", "away_9_y", "away_10_x", "away_10_y", "away_11_x", "away_11_y", 
        "away_12_x", "away_12_y", "away_13_x", "away_13_y",
        "speed_home_0", "speed_home_0", "speed_home_1", "speed_home_1", "speed_home_2", "speed_home_2", "speed_home_3", "speed_home_3", "speed_home_4", "speed_home_4", "speed_home_5", "speed_home_5", 
        "speed_home_6", "speed_home_6", "speed_home_7", "speed_home_7", "speed_home_8", "speed_home_8", "speed_home_9", "speed_home_9", "speed_home_10", "speed_home_10", "speed_home_11", "speed_home_11", 
        "speed_home_12", "speed_home_12", "speed_home_13", "speed_home_13",
        "speed_away_0", "speed_away_0", "speed_away_1", "speed_away_1", "speed_away_2", "speed_away_2", "speed_away_3", "speed_away_3", "speed_away_4", "speed_away_4", "speed_away_5", "speed_away_5", 
        "speed_away_6", "speed_away_6", "speed_away_7", "speed_away_7", "speed_away_8", "speed_away_8", "speed_away_9", "speed_away_9", "speed_away_10", "speed_away_10", "speed_away_11", "speed_away_11", 
        "speed_away_12", "speed_away_12", "speed_away_13", "speed_away_13",
        "acceleration_home_0", "acceleration_home_0", "acceleration_home_1", "acceleration_home_1", "acceleration_home_2", "acceleration_home_2", "acceleration_home_3", "acceleration_home_3", "acceleration_home_4", "acceleration_home_4", "acceleration_home_5", "acceleration_home_5", 
        "acceleration_home_6", "acceleration_home_6", "acceleration_home_7", "acceleration_home_7", "acceleration_home_8", "acceleration_home_8", "acceleration_home_9", "acceleration_home_9", "acceleration_home_10", "acceleration_home_10", "acceleration_home_11", "acceleration_home_11", 
        "acceleration_home_12", "acceleration_home_12", "acceleration_home_13", "acceleration_home_13",
        "acceleration_away_0", "acceleration_away_0", "acceleration_away_1", "acceleration_away_1", "acceleration_away_2", "acceleration_away_2", "acceleration_away_3", "acceleration_away_3", "acceleration_away_4", "acceleration_away_4", "acceleration_away_5", "acceleration_away_5", 
        "acceleration_away_6", "acceleration_away_6", "acceleration_away_7", "acceleration_away_7", "acceleration_away_8", "acceleration_away_8", "acceleration_away_9", "acceleration_away_9", "acceleration_away_10", "acceleration_away_10", "acceleration_away_11", "acceleration_away_11", 
        "acceleration_away_12", "acceleration_away_12", "acceleration_away_13", "acceleration_away_13",
    ]
}

hyperparameter_lstm = {
    "hyperparameters_1":{
        "hidden_dim":50,
        "num_layers":2,
        "epochs":50,
        "batch_size":64,
        "learning_rate":0.0001,
    }
}

length_field_in_metres = 105
width_field_in_metres = 68