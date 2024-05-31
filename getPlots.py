import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data_df = pd.read_csv("ObjectiveValues.csv")

# Plot settings
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

# Create a figure and axis
plt.figure(figsize=(12, 8))

# Plot optimal power versus w_fc
plt.plot(data_df["w_fc"], data_df["optimal_power"], marker='o', linestyle='-', color='b', label='Optimal Power (%BHP)')

# Plot optimal airspeed versus w_fc
plt.plot(data_df["w_fc"], data_df["optimal_airspeed"], marker='o', linestyle='-', color='g', label='Optimal Airspeed (kts)')

# Plot optimal fuel consumption versus w_fc
plt.plot(data_df["w_fc"], data_df["optimal_fc"], marker='o', linestyle='-', color='r', label='Optimal Fuel Consumption (gal/hr)')

# Add labels and title
plt.xlabel("$w_{\\mathrm{fuel\\_consumption}}$")
plt.ylabel("Values")
plt.title("Optimal Values vs. $w_{\\mathrm{fuel\\_consumption}}$", fontsize=16)

# Add grid
plt.grid(True)

# Add legend
plt.legend()

# Show the plot
plt.show()


from mpl_toolkits.mplot3d import Axes3D

# Read the CSV file
results_df = pd.read_csv("optimization_results.csv")

# Check the column names
print(results_df.columns)

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the data
sc = ax.scatter(results_df['optimal_power'], results_df['optimal_airspeed'], results_df['optimal_fc'], c='b', marker='o')

# Set the labels
ax.set_xlabel('Optimal Power (%BHP)')
ax.set_ylabel('Optimal Airspeed (kts)')
ax.set_zlabel('Optimal Fuel Consumption (gal/hr)')

# Show the plot
plt.show()