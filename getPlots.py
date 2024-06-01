import pandas as pd                     # type: ignore
import matplotlib.pyplot as plt         # type: ignore
import numpy as np                      # type: ignore
from mpl_toolkits.mplot3d import Axes3D # type: ignore
from scipy.spatial import cKDTree       # type: ignore

# Load data
results_df = pd.read_csv("computation/results.csv")
optima = pd.read_csv("computation/pareto_optima.csv")
optimum = pd.read_csv("computation/pareto_optimum.csv")

# Coordinates of non-Pareto/Pareto optimal points
non_pareto_pts = results_df[['optimal_power', 'optimal_airspeed', 'optimal_fc']].values
pareto_pts = optima[['optimal_power', 'optimal_airspeed', 'optimal_fc']].values

# Distances from each non-Pareto optimal point to the nearest Pareto optimal point
tree = cKDTree(pareto_pts)
distances, _ = tree.query(non_pareto_pts, k=1)

# Color gradient from light red to light pink based on distances
min_distance = min(distances)
max_distance = max(distances)
color_grad = np.linspace(1, 0.05, len(distances))
colors = plt.cm.Reds(color_grad)

# Create the figure and axis
fig = plt.figure(figsize=(7.5, 7.5))
ax = fig.add_subplot(111, projection='3d')

# Plot data points
sc = ax.scatter(results_df['optimal_power'], results_df['optimal_airspeed'], results_df['optimal_fc'], c=colors, marker='o', label='Non-Pareto')
sc_optima = ax.scatter(optima['optimal_power'], optima['optimal_airspeed'], optima['optimal_fc'], c='gold', marker='o', label='Pareto Optimal')
# sc_optimum = ax.scatter(optimum['optimal_power'], optimum['optimal_airspeed'], optimum['optimal_fc'], c='blue', marker='o', label=r"$\vec{\mathbf{w}} = (0.0,0.0,1.0)$", zorder=10)

# Set labels
ax.set_xlabel(r'Power (%BHP)')
ax.set_ylabel(r'Airspeed (kts)')
ax.set_zlabel(r'Fuel Consumption (gal/hr)')
ax.yaxis.label.set_position((0.5, -0.1))  # second value to move the label lower

ax.xaxis.pane.set_facecolor((0, 0.1, 0.7, 0.5))
ax.yaxis.pane.set_facecolor((0, 0.1, 0.7, 0.5))
ax.zaxis.pane.set_facecolor((0, 0.1, 0.7, 0.5))

ax.xaxis._axinfo["grid"].update(color = 'white', linestyle = '-', linewidth = 0.25)
ax.yaxis._axinfo["grid"].update(color = 'white', linestyle = '-', linewidth = 0.25)
ax.zaxis._axinfo["grid"].update(color = 'white', linestyle = '-', linewidth = 0.25)

ax.legend(fontsize='small', loc='upper left', bbox_to_anchor=(0.65, 0.8))

ax.view_init(elev=22.5, azim=210)
ax.set_proj_type("ortho")

plt.show()
