import pandas as pd

file_name = f"prediction_output/lstm_prediction_202502060033.pkl"

df_output_match4 = pd.read_pickle(file_name)

print(df_output_match4.describe())