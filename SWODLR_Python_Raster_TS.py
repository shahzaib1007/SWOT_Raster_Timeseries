import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from datetime import datetime
from matplotlib.animation import FuncAnimation
from pyproj import CRS
import geopandas as gpd
from shapely.geometry import box
from matplotlib.colors import Normalize
from tqdm import tqdm
import cartopy.io.img_tiles as cimgt
import argparse
from IPython.display import HTML
import rioxarray
from pathlib import Path
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")


def _process_file(file_path, gdf, output_dir):
    """Helper function to process one file for parallelization."""
    ds = xr.open_dataset(file_path).astype("float32")

    # CRS handling
    crs = CRS.from_wkt(ds.crs.attrs["crs_wkt"])
    gdf_proj = gdf.to_crs(crs.to_epsg())

    # Filtering by quantiles and flags
    q_low, q_high = np.float32(ds["wse"].quantile([0.05, 0.95]))
    ds = ds.where(
        (ds["wse"] >= q_low)
        & (ds["wse"] <= q_high)
        & (ds["dark_frac"] <= 0.3)
        & (ds["wse_qual"] <= 2),
        drop=True,
    )

    # Add time coordinate
    # date_obj = datetime.strptime(
    #     ds.attrs["time_granule_start"].split("T")[0], "%Y-%m-%d"
    # )
    date_obj = datetime.strptime(
        ds.attrs["time_granule_start"], "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    ds.coords["time"] = date_obj

    # Clip
    ds = ds.rio.write_crs(ds.crs.attrs["crs_wkt"])
    wse = ds[["wse"]].rio.clip(gdf_proj.geometry, gdf_proj.crs)

    # Build output path
    cycle = str(int(ds.attrs["cycle_number"])).zfill(3)
    pas = str(int(ds.attrs["pass_number"])).zfill(3)
    scene = str(int(ds.attrs["scene_number"])).zfill(3)
    date_str = date_obj.strftime("%Y%m%d")

    output_path = os.path.join(
        output_dir, f"SWOT_{date_str}_{cycle}_{pas}_{scene}_wse.nc"
    )
    wse.to_netcdf(output_path)

    return output_path

def filter_and_save(save_data_loc, clip_shp=None, bbox=None, max_workers=None):
    """Filter SWOT NetCDF files and save clipped versions inside Filtered_Data/.
       Either clip_shp or bbox must be provided.
    """

    if clip_shp:
        gdf = gpd.read_file(clip_shp)
    elif bbox:
        minx, miny, maxx, maxy = bbox
        gdf = gpd.GeoDataFrame(
            geometry=[box(minx, miny, maxx, maxy)], crs="EPSG:4326"
        )
    else:
        raise ValueError("Either shapefile path or bbox must be provided.")

    output_dir = os.path.join(save_data_loc, "Filtered_Data")
    os.makedirs(output_dir, exist_ok=True)
    save_data_loc_folder = Path(save_data_loc).joinpath("Downloaded_Data")
    os.makedirs(save_data_loc_folder, exist_ok=True)
    nc_files = [
        os.path.join(save_data_loc_folder, f)
        for f in os.listdir(save_data_loc_folder)
        if f.endswith(".nc")
    ]

    filtered_files = []
    max_workers = max_workers or (os.cpu_count() - 1)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_file, f, gdf, output_dir): f
            for f in nc_files
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Filtering files"):
            try:
                result = future.result()
                filtered_files.append(result)
            except Exception as e:
                print(f"Failed for {futures[future]}: {e}")

    return filtered_files


def concat_filtered(filtered_files, save_data_loc):
    """Concatenate filtered SWOT NetCDF files into one file."""
    datasets = [xr.open_dataset(f) for f in filtered_files]
    target_crs = datasets[0].rio.crs
    datasets = [
        d.rio.reproject(target_crs) if d.rio.crs != target_crs else d for d in datasets
    ]
    combined = xr.concat(datasets, dim="time")
    output_file = os.path.join(save_data_loc, "Filtered_Data", "combined_data.nc")
    combined.to_netcdf(output_file)
    return output_file


def create_swot_animation(
    input_nc,
    output_gif,
    fps=1,
    dpi=300,
    figsize=(12, 8),
    cmap="Blues",
    wse_threshold=0.01,
    overlay_path=None,
    bbox=None,
    basemap="satellite"
):
    """Create animation from SWOT data."""

    ds = xr.open_dataset(input_nc).sortby("time")
    ds["wse"] = ds["wse"].where(np.abs(ds["wse"]) > wse_threshold)

    crs_wkt = ds.crs.attrs["crs_wkt"]
    utm_crs = CRS.from_wkt(crs_wkt)
    utm_proj = ccrs.UTM(zone=utm_crs.to_dict()["zone"])
    geo_proj = ccrs.PlateCarree()

    if basemap == "satellite":
        tiler = cimgt.GoogleTiles(style="satellite")
    else:
        tiler = cimgt.Stamen("terrain-background")

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.8], projection=utm_proj)
    cax = fig.add_axes([0.82, 0.1, 0.02, 0.8])
    ax.add_image(tiler, 13)
    vmin, vmax = np.nanpercentile(ds["wse"], [5, 95])
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Water Surface Elevation (m)")

    # def add_basemap():
    #     ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
    #     ax.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=0)
    #     ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=1)
    #     ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5, zorder=1)
    #     gl = ax.gridlines(
    #         crs=geo_proj, draw_labels=True, linewidth=0.5, color="gray", alpha=0.5
    #     )
    #     gl.top_labels = False
    #     gl.right_labels = False

    def plot_overlay():
        if overlay_path:
            gdf = gpd.read_file(overlay_path).to_crs(utm_crs.to_string())
            for geom in gdf.geometry:
                ax.add_geometries([geom], crs=utm_proj, edgecolor="red", facecolor="none")
        elif bbox:
            minx, miny, maxx, maxy = bbox
            bbox_geom = gpd.GeoSeries(
                [box(minx, miny, maxx, maxy)], crs="EPSG:4326"
            ).to_crs(utm_crs.to_string())
            for geom in bbox_geom:
                ax.add_geometries([geom], crs=utm_proj, edgecolor="red", facecolor="none")

    def animate(i):
        for coll in ax.collections:  # remove only SWOT data
            coll.remove()
        # add_basemap()

        plot_overlay()
        im = ax.pcolormesh(
            ds.x,
            ds.y,
            ds["wse"].isel(time=i),
            cmap=cmap,
            norm=norm,
            transform=utm_proj,
            alpha=0.8,
        )
        time_str = str(ds.time[i].values)[:10]
        ax.set_title(f"SWOT Water Surface Elevation\n{time_str}")
        flood_annotation = ax.annotate(
            "",  # initially empty
            xy=(0.5, 0.95),
            xycoords='axes fraction',
            ha='center',
            fontsize=16,
            color='black',
            weight='bold',
            bbox=dict(boxstyle="round,pad=0.3",
                    fc="white",
                    ec="red",
                    lw=2,
                    alpha=0.7),
            zorder=3
        )

        if i == 0:
            flood_annotation.set_text("Event STARTED")
        elif i == len(ds.time) - 1:
            flood_annotation.set_text("Event ENDED")
        else:
            flood_annotation.set_text("")
        return im

    # add_basemap()
    plot_overlay()
    # ax.set_extent([ds.x.min(), ds.x.max(), ds.y.min(), ds.y.max()], crs=utm_proj)
    buffer = 1000  # meters
    x_min, x_max = float(ds.x.min()), float(ds.x.max())
    y_min, y_max = float(ds.y.min()), float(ds.y.max())

    ax.set_extent([x_min - buffer, x_max + buffer,
                y_min - buffer, y_max + buffer],
                crs=utm_proj)

    ani = FuncAnimation(fig, animate, frames=len(ds.time), interval=1000 / fps, blit=False)
    ani.save(output_gif, writer="pillow", fps=fps, dpi=dpi)
    plt.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter, concatenate, and animate SWOT data")

    parser.add_argument("--save_data_loc", type=str, required=True, help="Folder with NetCDFs (also used for outputs)")
    parser.add_argument("--shapefile", type=str, help="Path to shapefile/GeoJSON for clipping")
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("MINX", "MINY", "MAXX", "MAXY"),
                        help="Bounding box coordinates if shapefile not provided")
    parser.add_argument("--output_name", type=str, default="SWOT_animation.gif", help="Output GIF filename")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second for GIF")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved animation")
    parser.add_argument("--cmap", type=str, default="Blues", help="Colormap name")
    parser.add_argument("--max_workers", type=int, default=None, help="Number of parallel workers (default: CPU-1)")

    args = parser.parse_args()

    filtered_files = filter_and_save(args.save_data_loc, args.shapefile, args.bbox, args.max_workers)
    combined_nc = concat_filtered(filtered_files, args.save_data_loc)

    create_swot_animation(
        combined_nc,
        os.path.join(args.save_data_loc, args.output_name),
        fps=args.fps,
        dpi=args.dpi,
        cmap=args.cmap,
        bbox=args.bbox,
        overlay_path=args.shapefile,
    )