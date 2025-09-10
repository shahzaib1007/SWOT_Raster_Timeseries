import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.animation import FuncAnimation
from pyproj import CRS
import geopandas as gpd
from shapely.geometry import box
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
from tqdm import tqdm
import cartopy.io.img_tiles as cimgt
import argparse
from IPython.display import HTML
import pyproj
import warnings
warnings.filterwarnings("ignore")

def create_swot_animation(
    save_data_loc,
    output_name="SWOT_animation.gif",
    fps=1,
    dpi=300,
    figsize=(12, 8),
    cmap="Blues",
    wse_threshold=0.01,
    overlay_path=None,
    bbox=None,
    basemap="satellite"
):
    """Create animation from SWOT data with user-defined or default parameters."""

    input_nc = os.path.join(save_data_loc, "Filtered_Data", "combined_data.nc")
    output_gif = os.path.join(save_data_loc, output_name)

    print("Loading and preparing data...")
    ds = xr.open_dataset(input_nc)
    ds = ds.sortby('time')

    # Apply WSE threshold (values near zero treated as NaN)
    ds['wse'] = ds['wse'].where(np.abs(ds['wse']) > wse_threshold)
    
    crs_wkt = ds.crs.attrs['crs_wkt']
    utm_crs = CRS.from_wkt(crs_wkt)
    utm_proj = ccrs.UTM(zone=utm_crs.to_dict()['zone'])
    geo_proj = ccrs.PlateCarree()

    if basemap == "satellite":
        tiler = cimgt.QuadtreeTiles()  # works with e.g. GoogleTiles
    else:
        tiler = cimgt.Stamen("terrain-background")

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.8], projection=utm_proj)
    cax = fig.add_axes([0.82, 0.1, 0.02, 0.8])
    ax.add_image(tiler, 13)
    vmin, vmax = np.nanpercentile(ds['wse'], [5, 95])
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Water Surface Elevation (m)')

    def add_basemap(ax):
        """Add standard cartopy features to map."""
        ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='lightgray', zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='lightblue', zorder=0)
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5, zorder=1)
        ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':', linewidth=0.5, zorder=1)
        
        gl = ax.gridlines(crs=geo_proj, draw_labels=True,
                         linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False

    def plot_overlay(ax):
        """Plot shapefile/GeoJSON or bbox if provided."""
        if overlay_path:
            gdf = gpd.read_file(overlay_path)
            # Reproject shapefile/GeoJSON to UTM of NetCDF
            gdf = gdf.to_crs(utm_crs.to_string())
            for geom in gdf.geometry:
                ax.add_geometries([geom], crs=utm_proj,
                                edgecolor='red', facecolor='none', linewidth=2, zorder=4)
        else:
            # Default bbox in EPSG:4326 if not provided
            if bbox is None:
                # minx, miny, maxx, maxy
                bbox_4326 = [-180, -90, 180, 90]
            else:
                bbox_4326 = bbox
            minx, miny, maxx, maxy = bbox_4326
            # Transform the bbox to UTM
            bbox_geom = gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs="EPSG:4326").to_crs(utm_crs.to_string())
            for geom in bbox_geom:
                ax.add_geometries([geom], crs=utm_proj, edgecolor='red', facecolor='none', linewidth=2, zorder=4)

    flood_annotation = None

    def animate(i):
        """Update function for animation frames."""
        nonlocal flood_annotation
        
        if hasattr(ax, 'current_image'):
            ax.current_image.remove()
        
        if flood_annotation is not None:
            flood_annotation.remove()
            flood_annotation = None
        
        time_str = str(ds.time[i].values)[:10]
        
        im = ax.pcolormesh(
            ds.x, ds.y, ds['wse'].isel(time=i),
            cmap=cmap,
            norm=norm,
            transform=utm_proj,
            zorder=2,
            alpha=0.8
        )
        ax.current_image = im
        
        if i == 0:
            annotation_text = "Event STARTED"
        elif i == len(ds.time) - 1:
            annotation_text = "Event ENDED"
        else:
            annotation_text = ""
            
        if annotation_text:
            flood_annotation = ax.annotate(
                annotation_text,
                xy= (0.5, 0.95),
                xycoords='axes fraction',
                fontsize=16,
                color='black',
                weight='bold',
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="white",
                          ec="black",
                          lw=2,
                          alpha=0.7),
                zorder=3
            )
        
        ax.set_title(f"SWOT Water Surface Elevation\n{time_str}")
        return im

    # add_basemap(ax)
    plot_overlay(ax)
    # ax.set_extent([ds.x.min(), ds.x.max(), ds.y.min(), ds.y.max()], crs=utm_proj)
    buffer = 1000  # meters
    x_min, x_max = float(ds.x.min()), float(ds.x.max())
    y_min, y_max = float(ds.y.min()), float(ds.y.max())

    ax.set_extent([x_min - buffer, x_max + buffer,
                y_min - buffer, y_max + buffer],
                crs=utm_proj)

    print("Generating animation frames...")
    frames = len(ds.time)
    
    with tqdm(total=frames, desc="Rendering animation") as pbar:
        def update_with_progress(i):
            pbar.update(1)
            return animate(i)
        
        ani = FuncAnimation(
            fig=fig,
            func=update_with_progress,
            frames=frames,
            interval=1000/fps,  
            blit=False
        )
        
        print(f"Saving animation to {output_gif}...")
        ani.save(
            output_gif,
            writer='pillow',
            fps=fps,
            dpi=dpi,
            progress_callback=lambda i, n: pbar.update(0) 
        )
    
    plt.close()
    print("Animation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create SWOT WSE Animation")

    parser.add_argument("--save_data_loc", type=str, default="EarthData_SWOT/", help="Base folder where data is saved")
    parser.add_argument("--output_name", type=str, default="SWOT_animation.gif", help="Output GIF filename")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved animation")
    parser.add_argument("--figsize", nargs=2, type=float, default=[12, 8], help="Figure size in inches, e.g. --figsize 12 8")
    parser.add_argument("--cmap", type=str, default="Blues", help="Matplotlib colormap name")
    parser.add_argument("--wse_threshold", type=float, default=0.01, help="WSE threshold to filter near-zero values")
    parser.add_argument("--overlay", type=str, default=None, help="Path to shapefile or GeoJSON for overlay")
    parser.add_argument("--bbox", nargs=4, type=float, default=None, help="Bounding box to overlay: minx miny maxx maxy")

    args = parser.parse_args()

    create_swot_animation(
        save_data_loc=args.save_data_loc,
        output_name=args.output_name,
        fps=args.fps,
        dpi=args.dpi,
        figsize=tuple(args.figsize),
        cmap=args.cmap,
        wse_threshold=args.wse_threshold,overlay_path=args.overlay,
        bbox=args.bbox
    )
