"""
Flood Cluster Tracking from SWOT WSE NetCDF with Basemap and Animation
---------------------------------------------------------------------

Tracks multiple flood patches (connected components) across time
and assigns stable IDs using overlap-based matching.

Outputs:
  - NetCDF with cluster IDs
  - Animation of flood evolution
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, Normalize
from scipy.ndimage import label
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import geopandas as gpd
from shapely.geometry import box
from pyproj import CRS
from tqdm import tqdm
import argparse
from collections import defaultdict


# -------------------------------
# Cluster tracking
# -------------------------------
def track_clusters(flood_data, threshold=0.01):
    ntime, ny, nx = flood_data.shape
    tracked_labels = np.zeros_like(flood_data, dtype=int)
    next_id = 1
    prev_labels = None
    prev_ids = {}

    for t in range(ntime):
        mask = flood_data[t] > threshold
        labeled, ncomp = label(mask)
        current_ids = {}

        if prev_labels is None:
            # First timestep → every blob is new
            for comp_id in range(1, ncomp + 1):
                current_ids[comp_id] = next_id
                tracked_labels[t][labeled == comp_id] = next_id
                next_id += 1
        else:
            used_parents = set()
            for comp_id in range(1, ncomp + 1):
                comp_mask = (labeled == comp_id)
                overlap = np.bincount(prev_labels[comp_mask].ravel())
                if len(overlap) > 1:
                    best_prev = overlap[1:].argmax() + 1
                    if overlap[best_prev] > 0:
                        parent_id = prev_ids.get(best_prev, None)
                        if parent_id is not None:
                            if best_prev not in used_parents:
                                # Largest child keeps parent’s ID
                                match_id = parent_id
                                used_parents.add(best_prev)
                            else:
                                # Parent already assigned → new ID (split case)
                                match_id = next_id
                                next_id += 1
                        else:
                            match_id = next_id
                            next_id += 1
                    else:
                        match_id = next_id
                        next_id += 1
                else:
                    match_id = next_id
                    next_id += 1

                current_ids[comp_id] = match_id
                tracked_labels[t][comp_mask] = match_id

        prev_labels = labeled
        prev_ids = current_ids

    return tracked_labels


# -------------------------------
# Animation
# -------------------------------
def animate_clusters(
    ds, tracked_labels, out_file, interval=500,
    basemap="satellite", overlay=None, bbox=None,
    figsize=(12,8), cmap="tab20", store_frame=False,
    save_data_loc=None
):
    ntime = tracked_labels.shape[0]
    unique_ids = np.unique(tracked_labels)
    unique_ids = unique_ids[unique_ids != 0]
    vmax = tracked_labels.max()

    # Projection
    if "crs" in ds and "crs_wkt" in ds.crs.attrs:
        crs_wkt = ds.crs.attrs["crs_wkt"]
        utm_crs = CRS.from_wkt(crs_wkt)
        utm_proj = ccrs.UTM(zone=utm_crs.to_dict()["zone"])
    else:
        utm_proj = ccrs.PlateCarree()
    # geo_proj = ccrs.PlateCarree()

    # Basemap
    tiler = cimgt.QuadtreeTiles() if basemap=="satellite" else cimgt.Stamen("terrain-background")

    # Figure setup
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.1,0.1,0.7,0.8], projection=utm_proj)
    # cax = fig.add_axes([0.82,0.1,0.02,0.8])
    ax.add_image(tiler, 13)

    # Colormap
    base_cmap = plt.get_cmap(cmap)
    n_colors = max(unique_ids.max() + 1, base_cmap.N)
    colors = base_cmap(np.arange(n_colors) % base_cmap.N)
    custom_cmap = ListedColormap(np.vstack(([1,1,1,1], colors)))  
    # sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=Normalize(vmin=0, vmax=vmax))
    # cbar = fig.colorbar(sm, cax=cax)
    # cbar.set_label("Cluster ID")

    # Overlay
    def plot_overlay(ax):
        if overlay:
            gdf = gpd.read_file(overlay).to_crs(utm_proj)
            for geom in gdf.geometry:
                ax.add_geometries([geom], crs=utm_proj, edgecolor="red", facecolor="none", linewidth=2, zorder=4)
        else:
            if bbox is None:
                bbox_geom = gpd.GeoSeries([box(-180,-90,180,90)], crs="EPSG:4326").to_crs(utm_proj)
            else:
                minx, miny, maxx, maxy = bbox
                bbox_geom = gpd.GeoSeries([box(minx,miny,maxx,maxy)], crs="EPSG:4326").to_crs(utm_proj)
            for geom in bbox_geom:
                ax.add_geometries([geom], crs=utm_proj, edgecolor="red", facecolor="none", linewidth=2, zorder=4)
    plot_overlay(ax)
    
    if store_frame and save_data_loc is not None:
        frame_dir = os.path.join(save_data_loc, "png_frame")
        os.makedirs(frame_dir, exist_ok=True)

    flood_annotation = None

    # Update function
    def update(i):
        nonlocal flood_annotation
        for coll in ax.collections[:]:
            coll.remove()
        if flood_annotation is not None:
            flood_annotation.remove()
            flood_annotation = None

        arr = tracked_labels[i]
        arr_masked = np.ma.masked_where(arr==0, arr)

        im = ax.pcolormesh(
            ds.x, ds.y, arr_masked,
            cmap=custom_cmap, vmin=0, vmax=vmax,
            transform=utm_proj, zorder=2, alpha=0.8
        )

        time_val = ds.time[i].values
        if isinstance(time_val, np.datetime64):
            timestamp = pd.to_datetime(time_val)
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        else:
            timestamp_str = str(time_val)

        # Annotation
        if i==0:
            annotation_text="Event STARTED"
        elif i==ntime-1:
            annotation_text="Event ENDED"
        else:
            annotation_text=""
        if annotation_text:
            flood_annotation = ax.annotate(
                annotation_text, xy=(0.5,0.95), xycoords="axes fraction",
                fontsize=16, color="black", weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=2, alpha=0.7),
                zorder=3
            )

        if store_frame and save_data_loc is not None:
            # frame_file = os.path.join(frame_dir, f"frame_{i:03d}.png")
            # fig.savefig(frame_file, dpi=200, bbox_inches='tight')
            if isinstance(time_val, np.datetime64):
                timestamp = pd.to_datetime(time_val)
                filename_time = timestamp.strftime("%Y%m%d_%H%M%S")
                # frame_file = os.path.join(frame_dir, f"frame_{filename_time}.png")
                frame_file = os.path.join(frame_dir, f"frame_{i:03d}.png")
            else:
                # Fallback if not datetime
                frame_file = os.path.join(frame_dir, f"frame_{i:06d}.png")
            
            fig.savefig(frame_file, dpi=200, bbox_inches='tight')

        # ax.set_title(f"Flood Clusters | Time: {str(ds.time[i].values)[:10]}")
        ax.set_title(f"Flood Clusters | Time: {timestamp_str}")
        return [im]

    # Set extent
    buffer = 1000
    x_min, x_max = float(ds.x.min()), float(ds.x.max())
    y_min, y_max = float(ds.y.min()), float(ds.y.max())
    ax.set_extent([x_min-buffer, x_max+buffer, y_min-buffer, y_max+buffer], crs=utm_proj)

    print("Generating animation frames...")
    with tqdm(total=ntime, desc="Rendering animation") as pbar:
        def update_with_progress(i):
            pbar.update(1)
            return update(i)
        ani = FuncAnimation(fig, update_with_progress, frames=ntime, interval=interval, blit=False)
        ext = os.path.splitext(out_file)[1].lower()
        if ext==".gif":
            ani.save(out_file, writer="pillow", fps=1000/interval)
        else:
            ani.save(out_file, writer="ffmpeg", fps=1000/interval)
    plt.close()
    print(f"Animation saved -> {out_file}")

    if store_frame:
        print(f"Frames saved in -> {frame_dir}")

# -------------------------------
# Main function
# -------------------------------
def track_and_animate(
    input_folder, variable="wse", threshold=0.01,
    out_nc="flood_clusters.nc", out_anim="flood_clusters.gif",
    interval=500, basemap="satellite", overlay=None, bbox=None,
    figsize=(12,8), cmap="tab20", store_frame=False
):
    """Load NetCDF, track clusters, save NetCDF and animation."""
    input_nc = os.path.join(input_folder, "Filtered_Data", "combined_data.nc")
    if not os.path.exists(input_nc):
        raise FileNotFoundError(f"NetCDF file not found at {input_nc}")

    print("Loading data...")
    ds = xr.open_dataset(input_nc)
    ds = ds.sortby("time")
    if variable not in ds:
        raise ValueError(f"Variable '{variable}' not found. Available: {list(ds.data_vars)}")

    wse = ds[variable].values

    print("Tracking clusters...")
    tracked = track_clusters(wse, threshold=threshold)

    out_file = os.path.join(input_folder, out_nc)
    print(f"Saving NetCDF -> {out_file}")
    ds_out = xr.Dataset(
        {"cluster_id": (("time","y","x"), tracked)},
        coords={"time": ds.time, "y": ds.y, "x": ds.x}
    )
    ds_out.to_netcdf(out_file)

    out_gif_path = os.path.join(input_folder, out_anim)
    print(f"Saving animation -> {out_gif_path}")
    animate_clusters(
        ds, tracked, out_file=out_gif_path,
        interval=interval, basemap=basemap,
        overlay=overlay, bbox=bbox,
        figsize=figsize, cmap=cmap, store_frame=store_frame,
        save_data_loc=input_folder
    )

    print("Done.")


# -------------------------------
# CLI
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Track flood clusters from SWOT WSE NetCDF")
    parser.add_argument("--save_data_loc", required=True, help="Folder containing Filtered_Data/combined_data.nc")
    parser.add_argument("--variable", default="wse", help="Variable name in NetCDF")
    parser.add_argument("--threshold", type=float, default=0.01, help="Flood threshold")
    parser.add_argument("--out_nc", default="flood_clusters.nc", help="Output NetCDF")
    parser.add_argument("--out_anim", default="flood_clusters.gif", help="Output animation (.gif or .mp4)")
    parser.add_argument("--interval", type=int, default=500, help="Animation frame interval (ms)")
    parser.add_argument("--basemap", default="satellite", choices=["satellite","terrain-background"], help="Basemap type")
    parser.add_argument("--overlay", default=None, help="Shapefile/GeoJSON overlay")
    parser.add_argument("--bbox", nargs=4, type=float, default=None, help="Bounding box minx miny maxx maxy")
    parser.add_argument("--figsize", nargs=2, type=float, default=[12,8], help="Figure size inches")
    parser.add_argument("--cmap", default="tab20", help="Matplotlib colormap")
    parser.add_argument("--store_frame", action="store_true", help="Save each frame as PNG in png_frame folder")
    args = parser.parse_args()

    track_and_animate(
        input_folder=args.save_data_loc,
        variable=args.variable,
        threshold=args.threshold,
        out_nc=args.out_nc,
        out_anim=args.out_anim,
        interval=args.interval,
        basemap=args.basemap,
        overlay=args.overlay,
        bbox=args.bbox,
        figsize=tuple(args.figsize),
        cmap=args.cmap,
        store_frame=args.store_frame
    )


if __name__ == "__main__":
    main()
