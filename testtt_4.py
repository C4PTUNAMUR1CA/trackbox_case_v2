import pandas as pd
import numpy as np

# Creating a DataFrame with random float values
data = {
    'A': [1.1, 2.2, 3.3, 4.4],
    'B': [5.5, 6.6, 7.7, 8.8],
    'C': [9.9, 10.1, 11.2, 12.3]
}

df = pd.DataFrame(data)
df[["D","E"]] = [0,0]
df.loc[1,["D","E"]] = [1,1]
print(df[2:3])

predicted_cols = []
predicted_cols.append([0,0])
predicted_cols.append([1,1])

df[["ball_x_pred","ball_y_pred"]] = predicted_cols
lol=""