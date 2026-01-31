import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Constants
T_MAX_YEARS = 70.0
X_MAX_DIST = 18178.0
C_LOG_MAX = np.log10(212000 + 1)

source_coord = np.array([13278097, 279314])
target_coord = np.array([13286705, 298215])
unit_vec = (target_coord - source_coord) / np.linalg.norm(target_coord - source_coord)

class DioxanePINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
        self.v_raw = nn.Parameter(torch.tensor([0.0]))
        self.D_raw = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x, t):
        return self.network(torch.cat([x, t], dim=1))

@st.cache_resource
def load_model():
    model = DioxanePINN()
    model.load_state_dict(torch.load('model_130_170ft.pth', map_location='cpu', weights_only=True))
    model.eval()
    return model

st.title("🧪 Gelman 1,4-Dioxane Plume Forecaster")
st.markdown("PINN model predicting contaminant migration toward Barton Pond (130-170 ft depth)")

model = load_model()
year = st.slider("Select Year", 1986, 2056, 2024)

t_val = (year - 1986) / T_MAX_YEARS
x_grid = torch.linspace(0, 1, 500).view(-1, 1)
t_grid = torch.ones_like(x_grid) * t_val

with torch.no_grad():
    c_norm = model(x_grid, t_grid).numpy()

c_ppb = 10**(c_norm * C_LOG_MAX) - 1
x_feet = x_grid.numpy() * X_MAX_DIST

geo_x = source_coord[0] + (x_feet * unit_vec[0])
geo_y = source_coord[1] + (x_feet * unit_vec[1])

fig, ax = plt.subplots(figsize=(10, 8))
sc = ax.scatter(geo_x, geo_y, c=c_ppb.flatten(), cmap='jet', s=10,
                norm=LogNorm(vmin=1, vmax=100000))
ax.scatter(*source_coord, color='red', s=150, marker='X', zorder=5, label='Gelman Source')
ax.scatter(*target_coord, color='green', s=150, marker='*', zorder=5, label='Barton Pond')
ax.set_title(f"Predicted Plume — Year: {year}", fontsize=14, fontweight='bold')
ax.set_xlabel('Easting (ft)')
ax.set_ylabel('Northing (ft)')
ax.set_aspect('equal')
ax.legend(loc='upper left')
fig.colorbar(sc, ax=ax, label='1,4-Dioxane (ppb)')
st.pyplot(fig)

with torch.no_grad():
    c_pond = model(torch.tensor([[1.0]]), torch.tensor([[t_val]])).item()
    c_pond_ppb = 10**(c_pond * C_LOG_MAX) - 1

st.metric("Concentration at Barton Pond", f"{c_pond_ppb:.2f} ppb",
          delta="Safe" if c_pond_ppb < 7.2 else "Exceeds Limit",
          delta_color="normal" if c_pond_ppb < 7.2 else "inverse")
