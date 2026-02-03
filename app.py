import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import folium
from folium import plugins
from streamlit_folium import st_folium
from branca.element import Element
import os

# --- 1. Global Configuration & Constants ---
st.set_page_config(page_title="Gelman PINN Forecaster", layout="wide")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Physical Constants
X_MAX_DIST = 18178.0
T_MAX_YEARS = 70.0
C_LOG_MAX = np.log10(212000 + 1)
T_START_DATE = pd.Timestamp('1986-01-01')

# Geographic Anchors (Michigan State Plane / Lat Lon)
SOURCE_COORD_SPL = np.array([13278097, 279314])
TARGET_COORD_SPL = np.array([13286705, 298215])
SOURCE_LAT_LON = [42.2650, -83.7950]  # Wagner Rd Source
TARGET_LAT_LON = [42.3083, -83.7544]  # Barton Pond Intake

# --- 2. Model Architecture ---
class DioxanePINN(nn.Module):
    def __init__(self):
        super(DioxanePINN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
        self.v_raw = nn.Parameter(torch.tensor([0.0]))
        self.D_raw = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x_combined):
        return self.network(x_combined)

    def get_physics_params(self, unit_type="Unit E"):
        if "Unit E" in unit_type: # 130-170ft
            v = 0.46 + (0.96 - 0.46) * torch.sigmoid(self.v_raw)
            D = 5.47 + (21.90 - 5.47) * torch.sigmoid(self.D_raw)
        else: # Unit C3
            v = 0.41 + (0.68 - 0.41) * torch.sigmoid(self.v_raw)
            D = 2.73 + (13.68 - 2.73) * torch.sigmoid(self.D_raw)
        return v, D

# --- 3. Model Loading Helper ---
@st.cache_resource
def load_model(unit_selection):
    """Loads weights matching your GitHub repository structure."""
    model = DioxanePINN().to(device)
    
    if "Unit E" in unit_selection:
        filename = 'pinn_130_170_3input_final.pth'
    else:
        filename = 'weights_50_90ft_v2.pth'
    
    try:
        checkpoint = torch.load(filename, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model, filename
    except FileNotFoundError:
        return None, filename

# --- 4. Main Dashboard UI ---
st.title("🧪 Gelman 1,4-Dioxane Plume Forecaster")
st.markdown("""
**Physics-Informed Neural Network (PINN)** developed to forecast the migration of 1,4-Dioxane 
at the Gelman Sciences site in Ann Arbor, MI.
""")

# Sidebar Controls
st.sidebar.header("Model Configuration")
selected_unit = st.sidebar.selectbox("Select Aquifer Unit", ["Unit E (130-170ft)", "Unit C3 (50-90ft)"])
selected_year = st.sidebar.slider("Forecast Year", 1986, 2080, 2024)

# Load Model
model, loaded_filename = load_model(selected_unit)

if model:
    # Extract Physics Parameters
    v_tens, D_tens = model.get_physics_params(selected_unit)
    v_val = v_tens.item() * 365.25
    D_val = D_tens.item() * 365.25

    st.sidebar.markdown("---")
    st.sidebar.subheader(" calibrated physics")
    st.sidebar.metric("Learned Velocity (v)", f"{v_val:.1f} ft/yr")
    st.sidebar.metric("Learned Dispersion (D)", f"{D_val:.0f} ft²/yr")
    st.sidebar.success(f"Loaded: `{loaded_filename}`")

    # --- TABS FOR VISUALIZATION ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌍 Geographic Heatmap", 
        "🗺️ Spatiotemporal Evolution", 
        "🔍 Vertical Cross-Section", 
        "⛰️ 3D Topography"
    ])

    To fix the legend text visibility, I have added color: black; to the CSS style definition within the HTML string. This ensures the text renders in black regardless of the browser or Streamlit theme settings.

Here is the updated code for Tab 1 in your app.py.

📝 Updated app.py (Tab 1 Section)
Replace the with tab1: block with this version:

Python

    # === TAB 1: GEOGRAPHIC HEATMAP (FOLIUM) ===
    with tab1:
        st.subheader(f"Geographic Plume Spread ({selected_year})")
        
        # Inference Logic
        x_norm = torch.linspace(0, 1, 100, device=device).view(-1, 1)
        z_mid = torch.ones_like(x_norm) * 0.5
        t_val = (selected_year - 1986) / T_MAX_YEARS
        t_norm = torch.ones_like(x_norm) * t_val

        with torch.no_grad():
            c_out = model(torch.cat([x_norm, z_mid, t_norm], dim=1)).cpu().numpy().flatten()
        
        c_ppb = 10**(c_out * C_LOG_MAX) - 1

        # Interpolate Lat/Lon
        lats = np.linspace(SOURCE_LAT_LON[0], TARGET_LAT_LON[0], 100)
        lons = np.linspace(SOURCE_LAT_LON[1], TARGET_LAT_LON[1], 100)
        
        heat_data = [[lats[i], lons[i], float(c_ppb[i]/1000)] for i in range(100) if c_ppb[i] > 1.0]

        # Map Creation
        m = folium.Map(location=[42.288, -83.775], zoom_start=13, tiles='CartoDB positron')
        plugins.HeatMap(heat_data, radius=20, blur=15, min_opacity=0.3).add_to(m)

        # 7.2 ppb Front
        if len(c_ppb) > 0:
            front_idx = np.abs(c_ppb - 7.2).argmin()
            if c_ppb[front_idx] >= 1.0: 
                folium.Circle(
                    location=[lats[front_idx], lons[front_idx]],
                    radius=250, color='red', fill=True, fill_opacity=0.6,
                    popup=f'7.2 ppb Front ({selected_year})'
                ).add_to(m)

        # Markers
        folium.Marker(SOURCE_LAT_LON, popup="Gelman Source", icon=folium.Icon(color='red', icon='info-sign')).add_to(m)
        folium.Marker(TARGET_LAT_LON, popup="Barton Pond Intake", icon=folium.Icon(color='green', icon='leaf')).add_to(m)

        # Legend (FIXED: Added 'color: black;' to style)
        legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; width: 160px; height: 110px; 
             background-color: white; border:2px solid grey; z-index:9999; font-size:12px;
             padding: 10px; color: black;">
             <b>Plume Intensity</b><br>
             <i style="background: red; width: 10px; height: 10px; float: left; margin-right: 5px;"></i> High<br>
             <i style="background: yellow; width: 10px; height: 10px; float: left; margin-right: 5px;"></i> Medium<br>
             <i style="background: blue; width: 10px; height: 10px; float: left; margin-right: 5px;"></i> Low (>1.0 ppb)<br>
             <i style="border: 2px solid red; border-radius: 50%; width: 8px; height: 8px; float: left; margin-right: 5px;"></i> 7.2 ppb Front
        </div>
        '''
        m.get_root().html.add_child(Element(legend_html))

        # Render Map
        st_folium(m, width=800, height=500)
        
        # Impact Metric
        c_pond = c_ppb[-1].item()
        st.metric(
            "Concentration at Barton Pond Intake", 
            f"{c_pond:.2f} ppb",
            delta="Safe (<7.2 ppb)" if c_pond < 7.2 else "⚠️ EXCEEDS LIMIT",
            delta_color="normal" if c_pond < 7.2 else "inverse"
        )

    # === TAB 2: SPATIOTEMPORAL EVOLUTION (MATPLOTLIB) ===
    with tab2:
        st.subheader("Plume Evolution: 1986 vs 2024 vs Forecast")
        
        years_to_plot = [1986, 2024, 2060]
        fig2, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        
        x_res = 500
        x_grid = torch.linspace(0, 1, x_res, device=device).view(-1, 1)
        z_grid = torch.ones_like(x_grid) * 0.5 

        for i, yr in enumerate(years_to_plot):
            t_plot = torch.ones_like(x_grid) * ((yr - 1986) / T_MAX_YEARS)
            with torch.no_grad():
                # FIXED: Added .flatten() here as well for safe plotting
                c_out = model(torch.cat([x_grid, z_grid, t_plot], dim=1)).cpu().numpy().flatten()
            
            c_ppb_plot = 10**(c_out * C_LOG_MAX) - 1
            
            # Geo coords for plotting
            easting = SOURCE_COORD_SPL[0] + (x_grid.cpu().numpy().flatten() * (TARGET_COORD_SPL[0] - SOURCE_COORD_SPL[0]))
            northing = SOURCE_COORD_SPL[1] + (x_grid.cpu().numpy().flatten() * (TARGET_COORD_SPL[1] - SOURCE_COORD_SPL[1]))
            
            sc = axes[i].scatter(easting, northing, c=c_ppb_plot, cmap='jet', 
                               norm=LogNorm(vmin=1, vmax=1e5), s=20)
            
            axes[i].set_title(f"Year: {yr}", fontsize=12, fontweight='bold')
            axes[i].set_xlabel("Easting (ft)")
            axes[i].grid(True, alpha=0.3)
            
            # Markers
            axes[i].scatter(SOURCE_COORD_SPL[0], SOURCE_COORD_SPL[1], c='red', marker='X', s=100, label='Source')
            axes[i].scatter(TARGET_COORD_SPL[0], TARGET_COORD_SPL[1], c='green', marker='*', s=150, label='Intake')
        
        axes[0].set_ylabel("Northing (ft)")
        st.pyplot(fig2)

    # === TAB 3: VERTICAL CROSS-SECTION ===
    with tab3:
        st.subheader(f"Vertical Plume Profile: {selected_unit}")
        
        if "Unit E" in selected_unit:
            z_min, z_max = 130, 170
        else:
            z_min, z_max = 50, 90
            
        x_res, z_res = 200, 100
        x_mesh = np.linspace(0, 1, x_res)
        z_mesh = np.linspace(0, 1, z_res)
        X, Z = np.meshgrid(x_mesh, z_mesh)
        t_mesh = (selected_year - 1986) / T_MAX_YEARS
        
        input_tensor = torch.tensor(
            np.stack([X.ravel(), Z.ravel(), np.full(X.size, t_mesh)], axis=1), 
            dtype=torch.float32
        ).to(device)
        
        with torch.no_grad():
            c_pred = model(input_tensor).cpu().numpy().reshape(z_res, x_res)
        
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        x_feet = X * X_MAX_DIST
        z_feet = z_min + (Z * (z_max - z_min))
        
        heatmap = ax3.pcolormesh(x_feet, z_feet, c_pred, shading='gouraud', cmap='magma')
        fig3.colorbar(heatmap, label='Norm. Log-Concentration')
        ax3.invert_yaxis() 
        ax3.set_ylabel("Depth (ft)")
        ax3.set_xlabel("Distance from Source (ft)")
        ax3.set_title(f"Vertical Slice ({selected_year})")
        
        st.pyplot(fig3)

    # === TAB 4: 3D TOPOGRAPHY ===
    with tab4:
        st.subheader("3D Plume Landscape")
        
        # 1. Create layout columns to constrain the size
        # [1, 2, 1] creates empty space on left/right and puts plot in the middle 50%
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Re-using the mesh from Tab 3
            fig4 = plt.figure(figsize=(8, 6))
            ax4 = fig4.add_subplot(111, projection='3d')
            
            surf = ax4.plot_surface(x_feet, z_feet, c_pred, cmap='magma', edgecolor='none', alpha=0.85)
            
            ax4.set_xlabel('Distance (ft)', fontsize=8)
            ax4.set_ylabel('Depth (ft)', fontsize=8)
            ax4.set_zlabel('Log-Conc', fontsize=8)
            ax4.set_ylim(z_max, z_min)
            
            # Adjust tick label size for smaller plot
            ax4.tick_params(axis='both', which='major', labelsize=8)
            ax4.view_init(elev=35, azim=-60)
            
            plt.tight_layout()
            
            # 2. Render in the middle column
            st.pyplot(fig4, use_container_width=True)

else:
    st.error(f"⚠️ Model file not found. Expected: `{loaded_filename}`")
    st.info("Please make sure you have uploaded the .pth files to your GitHub repository.")
