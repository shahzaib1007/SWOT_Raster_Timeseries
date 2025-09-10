import os
import shutil
import argparse
import xarray as xr
import earthaccess
from datetime import datetime
import geopandas as gpd
import numpy as np
from ruamel.yaml import YAML
import rioxarray as rxr
from pyproj import CRS
from concurrent.futures import ProcessPoolExecutor, as_completed
from shapely.geometry import Polygon, box
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def check_bbox_intersection(granule, bbox):
    """Check if granule intersects with target bounding box."""
    try:
        spatial = granule['umm']['SpatialExtent']['HorizontalSpatialDomain']
        bounding_rects = spatial.get("Geometry", {}).get("BoundingRectangles", [])

        # Skip granules that are global (-180, -90, 180, 90)
        for rect in bounding_rects:
            if (rect.get("WestBoundingCoordinate") == -180 and
                rect.get("SouthBoundingCoordinate") == -90 and
                rect.get("EastBoundingCoordinate") == 180 and
                rect.get("NorthBoundingCoordinate") == 90):
                return False  # skip this granule

        # If polygon exists, check intersection with bbox
        if 'GPolygons' in spatial.get('Geometry', {}):
            points = spatial['Geometry']['GPolygons'][0]['Boundary']['Points']
            granule_poly = Polygon([(p['Longitude'], p['Latitude']) for p in points])
            bbox_poly = box(bbox[0], bbox[1], bbox[2], bbox[3])
            return bbox_poly.intersects(granule_poly)

        # If no polygon info, assume intersects (but not global bbox)
        return True
    except Exception:
        return True


def process_granule(args):
    """Process a single SWOT granule."""
    date_obj, granule, geojson, save_loc, bbox = args
    try:
        if not check_bbox_intersection(granule, bbox):
            return None

        # Download data
        url = granule.data_links()[0]
        native_id = granule['meta']['native-id']
        date_str = date_obj.strftime('%Y%m%d')
        granule_id = granule['umm']['RelatedUrls'][0]['URL'].split('x_x_x')[1].split('F')[0]
        local_path = f"{save_loc}/Downloaded_Data/SWOT_{date_str}_{granule_id}"
        os.makedirs(local_path, exist_ok=True)
        
        nc_file = f'{local_path}/{native_id}.nc'
        if not os.path.exists(nc_file):
            earthaccess.download(url, local_path=local_path)

        # Process data
        ds = xr.open_dataset(nc_file).astype('float32')
        crs = CRS.from_wkt(ds.crs.attrs['crs_wkt'])
        gdf = geojson.to_crs(crs.to_epsg())

        # Quality filters
        q_low, q_high = np.float32(ds['wse'].quantile([0.05, 0.95]))
        ds = ds.where(
            (ds['wse'] >= q_low) & 
            (ds['wse'] <= q_high) & 
            (ds['dark_frac'] <= 0.3) & 
            (ds['wse_qual'] <= 2),
            drop=True
        )

        # Add time and crs info
        ds.coords['time'] = date_obj
        ds = ds.rio.write_crs(ds.crs.attrs['crs_wkt'])
        
        # Clip and save
        wse = ds[['wse']].rio.clip(gdf.geometry, gdf.crs)
        output_path = f"{save_loc}/Filtered_Data/SWOT_{date_str}_{native_id.split('x_x_x_')[1].split('F')[0]}_wse.nc"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        wse.to_netcdf(output_path)
        ds.close()
        return output_path
    except Exception as e:
        return None


def main():
    args = parse_args()

    # Get bbox from shapefile if not given
    if args.bbox is not None:
        bbox = tuple(args.bbox)
    elif args.shapefile is not None:
        gdf = gpd.read_file(args.shapefile)
        bbox = tuple(gdf.total_bounds)  # xmin, ymin, xmax, ymax
    else:
        raise ValueError("You must provide either --bbox or --shapefile (shp or geojson)")

    save_loc = args.save_data_loc
    shutil.rmtree(f"{save_loc}/Filtered_Data/", ignore_errors=True)
    os.makedirs(save_loc, exist_ok=True)

    # Load shapefile for clipping
    geojson = gpd.read_file(args.shapefile) if args.shapefile else None

    # Login (default: netrc)
    earthaccess.login(strategy=args.login_strategy)

    print(f"Searching for {args.short_name} granules...")
    results = earthaccess.search_data(
        short_name=args.short_name,
        bounding_box=bbox,
        temporal=(args.start_date, args.end_date),
        granule_name=args.granule_name,
    )

    if not results:
        print("No granules found.")
        return

    print(f"Found {len(results)} granules. Processing with {args.workers} workers...")

    # Parallel processing
    tasks = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for granule in results:
            try:
                time_str = granule['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']
                date_obj = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
                tasks.append(executor.submit(process_granule, (date_obj, granule, geojson, save_loc, bbox)))
            except Exception:
                continue

        for f in tqdm(as_completed(tasks), total=len(tasks), desc="Processing granules"):
            _ = f.result()
    file_paths = [f.result() for f in tasks if f.result() is not None]
    # Combine results
    if file_paths:
        print("Combining results...")
        datasets = [xr.open_dataset(f) for f in file_paths]
        target_crs = datasets[0].rio.crs
        combined = xr.concat([d.rio.reproject(target_crs) if d.rio.crs != target_crs else d 
                            for d in datasets], dim='time')
        combined.to_netcdf(f"{save_loc}/Filtered_Data/combined_data.nc")
        print("Processing complete!")
    else:
        print("No valid data processed")

# def parse_args():
#     parser = argparse.ArgumentParser(description="Download and process SWOT WSE Raster data")

#     # Region args
#     parser.add_argument("--shapefile", type=str, default=None,
#                         help="Path to shapefile or GeoJSON file")
#     parser.add_argument("--bbox", nargs=4, type=float, default=None,
#                         metavar=("xmin", "ymin", "xmax", "ymax"),
#                         help="Bounding box: xmin ymin xmax ymax")

#     # Running config
#     parser.add_argument("--save_data_loc", type=str, default="../EarthData_SWOT/",
#                         help="Directory where data will be saved (default: ../EarthData_SWOT/)")
#     parser.add_argument("--start_date", type=str, 
#                         help="Start date (YYYY-MM-DD)")
#     parser.add_argument("--end_date", type=str, 
#                         help="End date (YYYY-MM-DD)")
#     default_workers = max(1, os.cpu_count() - 1)
#     parser.add_argument("--workers", type=int, default=default_workers,
#                         help=f"Number of parallel workers (default: {default_workers})")
#     parser.add_argument("--login_strategy", type=str, default="netrc",
#                         help="Earthaccess login strategy (default: netrc)")

#     # Dataset config
#     parser.add_argument("--short_name", type=str, default="SWOT_L2_HR_Raster_2.0",
#                         help="Dataset short name (default: SWOT_L2_HR_Raster_2.0)")
#     parser.add_argument("--granule_name", type=str, default="*_100m*PIC*_01",
#                         help="Granule name pattern (default: *_100m*PIC*_01)")

#     return parser.parse_args()

# if __name__ == "__main__":
#     main()

def parse_args(args_list=None):
    """Parse arguments, optionally from a list instead of command line"""
    parser = argparse.ArgumentParser(description="Download and process SWOT WSE Raster data")

    # Region args
    parser.add_argument("--shapefile", type=str, default=None,
                        help="Path to shapefile or GeoJSON file")
    parser.add_argument("--bbox", nargs=4, type=float, default=None,
                        metavar=("xmin", "ymin", "xmax", "ymax"),
                        help="Bounding box: xmin ymin xmax ymax")

    # Running config
    parser.add_argument("--save_data_loc", type=str, default="../EarthData_SWOT/",
                        help="Directory where data will be saved (default: ../EarthData_SWOT/)")
    parser.add_argument("--start_date", type=str, 
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, 
                        help="End date (YYYY-MM-DD)")
    default_workers = max(1, os.cpu_count() - 1)
    parser.add_argument("--workers", type=int, default=default_workers,
                        help=f"Number of parallel workers (default: {default_workers})")
    parser.add_argument("--login_strategy", type=str, default="netrc",
                        help="Earthaccess login strategy (default: netrc)")

    # Dataset config
    parser.add_argument("--short_name", type=str, default="SWOT_L2_HR_Raster_2.0",
                        help="Dataset short name (default: SWOT_L2_HR_Raster_2.0)")
    parser.add_argument("--granule_name", type=str, default="*_100m*PIC*_01",
                        help="Granule name pattern (default: *_100m*PIC*_01)")

    # Parse from list if provided, otherwise from command line
    if args_list is not None:
        return parser.parse_args(args_list)
    else:
        return parser.parse_args()


def EarthData_Automated_Download(shapefile=None, bbox=None, save_data_loc="../EarthData_SWOT/",
                               start_date=None, end_date=None, workers=None,
                               login_strategy="netrc", short_name="SWOT_L2_HR_Raster_2.0",
                               granule_name="*_100m*PIC*_01"):
    """
    Download and process SWOT WSE Raster data from NASA EarthData.
    
    Parameters:
    -----------
    shapefile : str, optional
        Path to shapefile or GeoJSON file defining the region of interest.
    bbox : list of float, optional
        Bounding box coordinates as [xmin, ymin, xmax, ymax].
    save_data_loc : str, default="../EarthData_SWOT/"
        Directory where downloaded and processed data will be saved.
    start_date : str
        Start date for data search in format 'YYYY-MM-DD'.
    end_date : str
        End date for data search in format 'YYYY-MM-DD'.
    workers : int, optional
        Number of parallel workers for processing.
    login_strategy : str, default="netrc"
        EarthData login strategy.
    short_name : str, default="SWOT_L2_HR_Raster_2.0"
        NASA EarthData dataset short name.
    granule_name : str, default="*_100m*PIC*_01"
        Granule name pattern for filtering specific data products.
    
    Returns:
    --------
    None
    """
    # Build command line arguments list
    args_list = []
    
    if shapefile:
        args_list.extend(["--shapefile", shapefile])
    if bbox:
        args_list.extend(["--bbox"] + [str(x) for x in bbox])
    if save_data_loc:
        args_list.extend(["--save_data_loc", save_data_loc])
    if start_date:
        args_list.extend(["--start_date", start_date])
    if end_date:
        args_list.extend(["--end_date", end_date])
    if workers:
        args_list.extend(["--workers", str(workers)])
    if login_strategy:
        args_list.extend(["--login_strategy", login_strategy])
    if short_name:
        args_list.extend(["--short_name", short_name])
    if granule_name:
        args_list.extend(["--granule_name", granule_name])
    
    # Parse the arguments
    args = parse_args(args_list)
    
    # Run the main function
    main(args)


# Keep this for command-line usage
if __name__ == "__main__":
    main()