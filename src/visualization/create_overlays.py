import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import geopandas as gpd
import xarray as xr
import numpy as np
import os
from shapely.geometry import Polygon, MultiPolygon

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Setup Dummy Data (Match evaluate.py Coords) ---
lats = np.linspace(3, 15, 48)
lons = np.linspace(33, 48, 60)

# Create dummy grid with range matching Bias (-2 to 2)
# This ensures colorbar ticks are identical (-2.0, -1.5, etc.)
data = np.linspace(-2.0, 2.0, len(lats) * len(lons)).reshape(len(lats), len(lons))
ds = xr.Dataset(
    {'dummy': (['lat', 'lon'], data)},
    coords={'lat': lats, 'lon': lons}
)

def generate_perfect_overlay(shapefile_path, output_name, line_color='black', line_width=1.0, dissolve=False):
    print(f"Generating aligned overlay: {output_name}...")
    
    # --- 2. Mimic evaluate.py Layout EXACTLY ---
    extent = [32.5, 48.5, 3, 15]
    common_cbar_kwargs = {'shrink': 0.7, 'pad': 0.05, 'aspect': 20}
    
    # We use the text from the "Spatial Bias" plot as the reference
    dummy_title = "Spatial Bias (Forecast - Obs) - 2020" 
    dummy_label = "Mean Bias (mm/day)"
    
    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_alpha(0.0) # Transparent Figure
    
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.patch.set_alpha(0.0) # Transparent Axes
    
    # --- LAYOUT ANCHORS (Updated to match evaluate.py) ---
    anchor_style = dict(fontsize=20, color='#00000001') # Alpha=0.001
    # fig.text(0.5, 1.05, "."*100, ha='center', **anchor_style) # Removed Top Anchor
    fig.text(0.94, 0.5, "."*60, va='center', rotation=270, **anchor_style) # Right (Tightened)
    
    # --- 3. Plot Invisible Heatmap ---
    bias_levels = [-2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0]
    
    plot_handle = ds['dummy'].plot(
        ax=ax, transform=ccrs.PlateCarree(),
        cmap='RdBu', 
        levels=bias_levels,
        extend='both',
        alpha=0.0,   # Invisible
        add_colorbar=True, 
        cbar_kwargs={**common_cbar_kwargs, 'label': dummy_label}
    )
    
    # Hide Colorbar
    cbar = plot_handle.colorbar
    cbar.outline.set_visible(False)
    
    # Use 0.001 alpha so bbox sees it (and matches base map)
    ghost_color = (0, 0, 0, 0.001)
    cbar.ax.yaxis.set_tick_params(color=ghost_color, labelcolor=ghost_color) 
    cbar.ax.yaxis.label.set_color(ghost_color) 
    cbar.solids.set_alpha(0) 

    # --- 5. Add Gridlines ---
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.0)
    gl.top_labels = False   # Turn off top labels
    gl.right_labels = False # Turn off right labels
    gl.xlabel_style = {'color': ghost_color} 
    gl.ylabel_style = {'color': ghost_color}
    
    # --- 6. Add Title ---
    plt.title(dummy_title, fontsize=14, pad=10, color=ghost_color)
    
    # --- 7. Add Shapefile ---
    try:
        gdf = gpd.read_file(shapefile_path)
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
            
        if dissolve:
            # 0. Buffer slightly to close tiny gaps between regions (Topology Fix)
            # This prevents internal lines from appearing where borders don't perfectly touch.
            gdf['geometry'] = gdf.geometry.buffer(0.001)

            # 1. Dissolve all features
            gdf = gdf.dissolve()
            
            # 2. Explode to handle potential disjoint parts (islands/slivers)
            gdf_exploded = gdf.explode(index_parts=False)
            
            # 3. Find the largest polygon (Main Landmass)
            # Degrees area is sufficient to distinguish Ethiopia from tiny slivers
            gdf_exploded['area'] = gdf_exploded.geometry.area
            largest_poly_row = gdf_exploded.sort_values('area', ascending=False).iloc[0]
            largest_geom = largest_poly_row.geometry
            
            # 4. Remove holes (Lake Tana, etc.)
            if largest_geom.geom_type == 'Polygon':
                clean_geom = Polygon(largest_geom.exterior)
            else:
                clean_geom = largest_geom
                
            # 5. Recreate GDF
            gdf = gpd.GeoDataFrame(geometry=[clean_geom], crs=gdf.crs)
        
        ax.add_geometries(gdf.geometry, ccrs.PlateCarree(),
                          facecolor='none', edgecolor=line_color, linewidth=line_width)
    except Exception as e:
        print(f"Error reading shapefile: {e}")
    
    # --- 8. Save ---
    ax.spines['geo'].set_visible(False) # Turn off boundary
    
    plt.savefig(os.path.join(OUTPUT_DIR, output_name), dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

if __name__ == "__main__":
    base_dir = r"c:\Users\yonas\Documents\S2S-Forecast-Project\data\shapefiles"
    regions = os.path.join(base_dir, "ETH_Regions.shp")
    zones = os.path.join(base_dir, "ETH_Zones.shp")
    
    if os.path.exists(regions):
        generate_perfect_overlay(regions, "overlay_regions.png", 'black', 1.5)
        # Generate Country Boundary Overlay
        generate_perfect_overlay(regions, "overlay_country.png", 'black', 2.0, dissolve=True)
        
    if os.path.exists(zones):
        generate_perfect_overlay(zones, "overlay_zones.png", 'black', 0.8)
