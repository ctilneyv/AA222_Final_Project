import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data_df = pd.read_csv("ObjectiveValues.csv")

# Plot settings
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Plot optimal power versus w_fuel_consumption
plt.figure(figsize=(10, 6))
plt.plot(data_df["w_fc"], data_df["optimal_power"], marker='o', linestyle='-', color='b')
plt.xlabel("$w_{\\mathrm{fuel\\_consumption}}$")
plt.ylabel("Optimal Power (\%BHP)")
plt.title("Optimal Power vs. $w_{\\mathrm{fuel\\_consumption}}$", fontsize=16)
plt.grid(True)
plt.show()

# Plot optimal airspeed versus w_fuel_consumption
plt.figure(figsize=(10, 6))
plt.plot(data_df["w_fc"], data_df["optimal_airspeed"], marker='o', linestyle='-', color='g')
plt.xlabel("$w_{\\mathrm{fuel\\_consumption}}$")
plt.ylabel("Optimal Airspeed (kts)")
plt.title("Optimal Airspeed vs. $w_{\\mathrm{fuel\\_consumption}}$", fontsize=16)
plt.grid(True)
plt.show()

# Plot optimal fuel consumption versus w_fuel_consumption
plt.figure(figsize=(10, 6))
plt.plot(data_df["w_fc"], data_df["optimal_fc"], marker='o', linestyle='-', color='r')
plt.xlabel("$w_{\\mathrm{fuel\\_consumption}}$")
plt.ylabel("Optimal Fuel Consumption (gal/hr)")
plt.title("Optimal Fuel Consumption vs. $w_{\\mathrm{fuel\\_consumption}}$", fontsize=16)
plt.grid(True)
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