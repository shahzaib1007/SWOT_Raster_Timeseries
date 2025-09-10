#!/usr/bin/env python3
# from ruamel.yaml import YAML
import argparse
import json, os
import re
import requests
import earthaccess
from datetime import datetime
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Polygon, box
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def get_swot_cps(config):
    """Search SWOT granules using YAML config"""
    auth = earthaccess.login(strategy=config.strategy)
    results = earthaccess.search_data(
        short_name=config.short_name,
        cloud_hosted=True,
        bounding_box=tuple(config.bbox),
        temporal=(config.start_date, config.end_date),
        granule_name=config.granule_filter,
    )

    cps_set = set()
    bbox_polygon = box(*config.bbox)

    for granule in tqdm(results, desc="Processing granules"):
        try:
            umm = granule.get("umm", {})
            granule_name = umm.get("GranuleUR", "")
            
            bounding_rects = umm.get("SpatialExtent", {}) \
                            .get("HorizontalSpatialDomain", {}) \
                            .get("Geometry", {}) \
                            .get("BoundingRectangles", [])
            skip_granule = False
            for rect in bounding_rects:
                if (rect.get("WestBoundingCoordinate") == -180 and
                    rect.get("SouthBoundingCoordinate") == -90 and
                    rect.get("EastBoundingCoordinate") == 180 and
                    rect.get("NorthBoundingCoordinate") == 90):
                    skip_granule = True
                    break
            if skip_granule:
                continue

            if config.check_intersection:
                if 'GPolygons' in umm['SpatialExtent']['HorizontalSpatialDomain']['Geometry']:
                    points = umm['SpatialExtent']['HorizontalSpatialDomain']['Geometry']['GPolygons'][0]['Boundary']['Points']
                    granule_polygon = Polygon([(point['Longitude'], point['Latitude']) for point in points])
                    if not bbox_polygon.intersects(granule_polygon):
                        continue
            
            match = re.search(r'_(\d{3})_(\d{3})_(\d{3}[A-Z]?)_', granule_name)
            if match:
                cycle, pass_num, scene = match.groups()
                date_str = umm['TemporalExtent']['RangeDateTime']['BeginningDateTime']
                date_obj = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
                cps_set.add((
                    int(cycle), 
                    int(pass_num), 
                    int(scene[:3]), 
                    date_obj.strftime("%Y-%m-%d")
                ))
        except Exception as e:
            print(f"Error processing {granule_name}: {e}")
    
    return sorted(cps_set)

def generate_products(cps_combos, config):
    """Generate products using user's input"""
    auth = earthaccess.login(strategy=config.strategy)
    token = auth.token['access_token']
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    output_file = Path(config.output_file)
    try:
        os.remove(output_file)
    except:
        pass
    output_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_file) as f:
            generated_products = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        generated_products = []
    
    existing_combos = {(p['cycle'], p['pass'], p['scene']) for p in generated_products}
    # raster_resolution = config['api'].get('raster_resolution', 100)
    for cycle, pass_num, scene, date in tqdm(cps_combos, desc="Requesting products"):
        if (cycle, pass_num, scene) in existing_combos:
            continue
        mutation = f"""
            mutation {{
                generateL2RasterProduct(
                    cycle: {cycle},
                    pass: {pass_num},
                    scene: {scene},
                    outputGranuleExtentFlag: true,
                    outputSamplingGridType: UTM,
                    rasterResolution: {config.raster_resolution},
                    utmZoneAdjust: 0,
                    mgrsBandAdjust: 0,
                ) {{
                    id
                    timestamp
                    cycle
                    pass
                    scene
                    status {{
                        timestamp
                        state
                    }}
                    granules {{
                        uri
                    }}
                }}
            }}
            """
        try:
            response = requests.post(
                config.api_url,
                headers=headers,
                json={"query": mutation},
                timeout=config.timeout
            )
            
            if response.ok:
                product = response.json()['data']['generateL2RasterProduct']
                generated_products.append({
                    'id': product['id'],
                    'cycle': cycle,
                    'pass': pass_num,
                    'scene': scene,
                    'date': date,
                    'status': product.get('status', {}),
                    'granules': product.get('granules', []),\
                })
                
                with open(output_file, 'w') as f:
                    json.dump(generated_products, f, indent=2)
                
                    
        except Exception as e:
            print(f"Error requesting {cycle}-{pass_num}-{scene}: {e}")
            # pass
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SWOT raster products using SWODLR API")

    # Required params
    parser.add_argument("--start_date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--raster_resolution", type=int, required=True, help="Raster resolution in meters")
    parser.add_argument("--shapefile", type=str, help="Path to shapefile (.shp) for bounding box")
    parser.add_argument("--geojson", type=str, help="Path to GeoJSON file for bounding box")

    # Defaults params (can still be overridden if desired)
    parser.add_argument("--strategy", type=str, default="netrc", help="Authentication strategy (default: netrc)")
    parser.add_argument("--short_name", type=str, default="SWOT_L2_HR_Raster_2.0", help="Product short name")
    parser.add_argument("--granule_filter", type=str, default="*100m*PGC0*", help="Granule filter")
    parser.add_argument("--api_url", type=str, default="https://swodlr.podaac.earthdatacloud.nasa.gov/api/graphql", help="SWODLR API URL")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    parser.add_argument("--output_file", type=str, default="SWODLR_generated_products.json", help="Path to output JSON file")
    parser.add_argument("--check_intersection", action="store_true", help="Check intersection with granule polygons")
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("MINX", "MINY", "MAXX", "MAXY"), help="Bounding box coordinates: minx miny maxx maxy (optional)")

    
    args = parser.parse_args()
    # handle bounding box from shp/geojson
    if args.bbox:
        args.bbox = tuple(args.bbox)
    elif args.shapefile or args.geojson:
        shp_path = args.shapefile or args.geojson
        gdf = gpd.read_file(shp_path)
        args.bbox = tuple(float(x) for x in gdf.total_bounds)
    else:
        raise ValueError("You must provide either --bbox, --shapefile, or --geojson for bounding box")

    cps_combos = get_swot_cps(args)
    print(f"Found {len(cps_combos)} valid CPS combos")
    generate_products(cps_combos, args)