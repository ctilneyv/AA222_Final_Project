import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data_df = pd.read_csv("ObjectiveValues.csv")

# Plot the objective values versus w_fuel_consumption
plt.plot(data_df["w_fuel_consumption"], data_df["objective_value"], marker='o', linestyle='-')
plt.xlabel("$w_{\mathrm{fuel\_consumption}}$")
plt.ylabel("Objective Value")
plt.title("Objective Values vs. $w_{\mathrm{fuel\_consumption}}$", fontsize=16)
plt.grid(True)
plt.show()