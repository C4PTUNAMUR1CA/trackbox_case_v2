import matplotlib.pyplot as plt
import pandas as pd

# Sample DataFrame
data = {
    'IdPeriod': [1, 1, 1, 2, 2, 2],  # First and second halves
    'Time': [0, 0.1, 0.2, 0, 0.1, 0.2],  # Time within each half
    'ball_x_pred': [10, 15, 20, 5, 8, 12],  # Predicted X coordinates
    'ball_y_pred': [30, 35, 40, 25, 28, 30]  # Predicted Y coordinates
}

df = pd.DataFrame(data)

# Create a plot
plt.figure(figsize=(8, 6))

# Plot trajectories for each period separately
for period in df['IdPeriod'].unique():
    subset = df[df['IdPeriod'] == period]
    plt.plot(subset['ball_x_pred'], subset['ball_y_pred'], marker='o', linestyle='-', label=f'Period {period}')

# Labels and title
plt.xlabel('Ball X Coordinate')
plt.ylabel('Ball Y Coordinate')
plt.title('Ball Trajectory')
plt.legend()
plt.grid(True)
plt.show()