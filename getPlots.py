import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import plotly.graph_objects as go
import plotlyshare

results_df = pd.read_csv("computation/results.csv")
optima = pd.read_csv("computation/pareto_optima.csv")
optimum = pd.read_csv("computation/pareto_optimum.csv")

non_pareto_pts = results_df[['optimal_power', 'optimal_airspeed', 'optimal_fc']].values
pareto_pts = optima[['optimal_power', 'optimal_airspeed', 'optimal_fc']].values

tree = cKDTree(pareto_pts)
distances, _ = tree.query(non_pareto_pts, k=1)

min_distance = min(distances)
max_distance = max(distances)
color_grad = np.linspace(1, 0.05, len(distances))
colors = plt.cm.Reds(color_grad)

fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=results_df['optimal_power'],
    y=results_df['optimal_airspeed'],
    z=results_df['optimal_fc'],
    mode='markers',
    marker=dict(
        size=3,
        color=distances,
        colorscale='Reds',
        colorbar=dict(title='Distance'),
    ),
    name='Non-Pareto'
))

fig.add_trace(go.Scatter3d(
    x=optima['optimal_power'],
    y=optima['optimal_airspeed'],
    z=optima['optimal_fc'],
    mode='markers',
    marker=dict(
        size=3,
        color='gold'
    ),
    name='Pareto Optimal'
))

fig.add_trace(go.Scatter3d(
    x=optimum['optimal_power'],
    y=optimum['optimal_airspeed'],
    z=optimum['optimal_fc'],
    mode='markers',
    marker=dict(
        size=5,
        color='blue'
    ),
    name=r"$\vec{\mathbf{w}} = (0.2,0.2,0.6)$"
))

fig.update_layout(
    scene=dict(
        xaxis_title='Power (%BHP)',
        yaxis_title='Airspeed (kts)',
        zaxis_title='Fuel Consumption (gal/hr)',
        xaxis=dict(backgroundcolor="rgb(0, 26, 102)", gridcolor="white", showbackground=True, zerolinecolor="white"),
        yaxis=dict(backgroundcolor="rgb(0, 26, 102)", gridcolor="white", showbackground=True, zerolinecolor="white"),
        zaxis=dict(backgroundcolor="rgb(0, 26, 102)", gridcolor="white", showbackground=True, zerolinecolor="white"),
    ),
    legend=dict(font=dict(size=10)),
    margin=dict(l=0, r=0, b=0, t=0)
)

fig.update_layout(
    scene_camera=dict(
        eye=dict(x=2.0, y=2.0, z=0.5)
    )
)

# fig.write_html("InteractivePareto.html")
# fig.show()
fig.show(renderer='plotlyshare')

"""import pandas as pd                     # type: ignore
import matplotlib.pyplot as plt         # type: ignore
import numpy as np                      # type: ignore
from mpl_toolkits.mplot3d import Axes3D # type: ignore
from scipy.spatial import cKDTree       # type: ignore

results_df = pd.read_csv("computation/results.csv")
optima = pd.read_csv("computation/pareto_optima.csv")
optimum = pd.read_csv("computation/pareto_optimum.csv")

# non-Pareto/Pareto optimal points
non_pareto_pts = results_df[['optimal_power', 'optimal_airspeed', 'optimal_fc']].values
pareto_pts = optima[['optimal_power', 'optimal_airspeed', 'optimal_fc']].values

tree = cKDTree(pareto_pts)
distances, _ = tree.query(non_pareto_pts, k=1)

# color gradient from light red to light pink based on distances
min_distance = min(distances)
max_distance = max(distances)
color_grad = np.linspace(1, 0.05, len(distances))
colors = plt.cm.Reds(color_grad)

fig = plt.figure(figsize=(7.5, 7.5))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(results_df['optimal_power'], results_df['optimal_airspeed'], results_df['optimal_fc'], c=colors, marker='o', label='Non-Pareto')
sc_optima = ax.scatter(optima['optimal_power'], optima['optimal_airspeed'], optima['optimal_fc'], c='gold', marker='o', label='Pareto Optimal')
# sc_optimum = ax.scatter(optimum['optimal_power'], optimum['optimal_airspeed'], optimum['optimal_fc'], c='blue', marker='o', label=r"$\vec{\mathbf{w}} = (0.0,0.0,1.0)$", zorder=10)

ax.set_xlabel(r'Power (%BHP)')
ax.set_ylabel(r'Airspeed (kts)')
ax.set_zlabel(r'Fuel Consumption (gal/hr)')
ax.yaxis.label.set_position((0.5, -0.1))

ax.xaxis.pane.set_facecolor((0, 0.1, 0.7, 0.5))
ax.yaxis.pane.set_facecolor((0, 0.1, 0.7, 0.5))
ax.zaxis.pane.set_facecolor((0, 0.1, 0.7, 0.5))

ax.xaxis._axinfo["grid"].update(color = 'white', linestyle = '-', linewidth = 0.25)
ax.yaxis._axinfo["grid"].update(color = 'white', linestyle = '-', linewidth = 0.25)
ax.zaxis._axinfo["grid"].update(color = 'white', linestyle = '-', linewidth = 0.25)

ax.legend(fontsize='small', loc='upper left', bbox_to_anchor=(0.65, 0.8))

ax.view_init(elev=17.5, azim=205)
ax.set_proj_type("ortho")

plt.show()"""