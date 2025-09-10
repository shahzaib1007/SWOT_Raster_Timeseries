import argparse
import json
import time
import os
import requests
from pathlib import Path
from tqdm import tqdm
from urllib.parse import urlparse
import earthaccess
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import warnings
warnings.filterwarnings("ignore")

def check_product_status(product_id, auth_token, api_url, max_checks=30, interval=60):
    """Check product status with retries"""
    # auth = earthaccess.login(strategy=config['auth']['strategy'])
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }
    status_history = []
    status_template = """
    query {
      l2RasterProduct(id: "%s") {
        status {
          timestamp
          state
          reason
        }
        granules {
          uri
        }
      }
    }
    """
    for attempt in range(max_checks):
        try:
            response = requests.post(
                api_url, headers=headers, json={"query": status_template % product_id}, timeout=30
            )
            response.raise_for_status()
            result = response.json()
            if 'errors' in result:
                print(f"GraphQL error: {result['errors']}")
                return None, status_history
            
                
            product_data = result['data']['l2RasterProduct']
            current_status = product_data['status'][0]  # Most recent status is first
            status_history.append(current_status)
            
            print(f"Attempt {attempt + 1}: Status {current_status['state']} at {current_status['timestamp']}")
            
            if current_status['state'] == "READY":
                return product_data.get('granules', []), status_history
            elif current_status['state'] in ["ERROR", "FAILED", "CANCELLED"]:
                return None, status_history
                
            time.sleep(interval)
            
        except requests.exceptions.RequestException as e:
            print(f"Network error checking status: {e}")
            status_history.append({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "state": "NETWORK_ERROR",
                "reason": str(e)
            })
            time.sleep(interval)
        except Exception as e:
            print(f"Unexpected error checking status: {e}")
            status_history.append({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "state": "CHECK_ERROR",
                "reason": str(e)
            })
            time.sleep(interval)
    
    return None, status_history

def download_file(url, dest_path, session, max_retries=3, retry_delay=10, timeout=300):
    """Download a single file with progress tracking and retries"""
    # headers = {"Authorization": f"Bearer {auth_token}"}
    
    for attempt in range(max_retries):
        try:
            with session.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                
                with open(dest_path, 'wb') as f, tqdm(
                    desc=os.path.basename(dest_path),
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    leave=False
                ) as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:  # filter out keep-alive chunks
                            f.write(chunk)
                            bar.update(len(chunk))
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    return False

# def download_products(args):
#     """Download products with comprehensive tracking and error handling"""
#     auth = earthaccess.login(strategy=args.strategy)
#     save_data_loc = Path(args.save_data_loc)
#     save_data_loc.mkdir(parents=True, exist_ok=True)
    
#     # Load existing products
#     try:
#         with open(args.input_file) as f:
#             products = json.load(f)
#     except (FileNotFoundError, json.JSONDecodeError) as e:
#         print(f"Error loading product file: {e}")
#         return
    
#     updated_products = []
#     successful_downloads = 0
    
#     for product in tqdm(products, desc="Processing products"):
#         if product.get('downloaded'):
#             updated_products.append(product)
#             continue
        
#         # Initialize tracking fields
#         product.setdefault('status_history', [])
#         product.setdefault('download_attempts', 0)
        
#         # Check product status
#         granules, status_history = check_product_status(product['id'],
#             auth.token['access_token'],
#             api_url=args.api_url,
#             max_checks=args.max_checks,
#             interval=args.check_interval)
#         product['status_history'].extend(status_history)
        
#         if not granules:
#             product['error'] = "Product not ready or failed"
#             updated_products.append(product)
#             continue
        
#         # Download granules
#         downloaded_files = []
#         for granule in granules:
#             url = granule['uri']
#             filename = os.path.basename(urlparse(url).path)
#             dest_path = save_data_loc.joinpath(filename)

#             # Skip if already downloaded
#             if dest_path.exists():
#                 downloaded_files.append(str(dest_path))
#                 continue
            
#             product['download_attempts'] += 1
#             if download_file(url, dest_path, auth.token['access_token'], args.max_retries, args.retry_delay, args.timeout):
#                 downloaded_files.append(str(dest_path))
#             else:
#                 print(f"Failed to download {filename} after {args.max_retries} attempts")
        
#         if downloaded_files:
#             product.update({
#                 'downloaded': True,
#                 'downloaded_at': time.strftime("%Y-%m-%dT%H:%M:%S"),
#                 'downloaded_files': downloaded_files,
#                 'final_state': 'SUCCESS'
#             })
#             successful_downloads += 1
#         else:
#             product['final_state'] = 'DOWNLOAD_FAILED'
        
#         updated_products.append(product)
        
#         # Save progress after each product
#         with open(args.input_file, 'w') as f:
#             json.dump(updated_products, f, indent=2)
    
#     print(f"\nDownload complete. Successfully downloaded {successful_downloads}/{len(products)} products")
    # print(f"Download directory: {save_data_loc.absolute()}")
def download_products(args):
    auth = earthaccess.login(strategy=args.strategy)
    save_data_loc = Path(args.save_data_loc)
    save_data_loc.mkdir(parents=True, exist_ok=True)
    dest_path_folder = save_data_loc.joinpath('Downloaded_Data')
    os.makedirs(dest_path_folder, exist_ok=True)
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {auth.token['access_token']}"})
    with open(args.input_file) as f:
        products = json.load(f)

    updated_products = []
    successful_downloads = 0
    max_workers = args.max_workers or max(1, (os.cpu_count() or 2) - 1)
    for product in tqdm(products, desc="Processing products"):
        if product.get('downloaded'):
            updated_products.append(product)
            continue

        granules, status_history = check_product_status(
            product['id'], auth.token['access_token'],
            api_url=args.api_url,
            max_checks=args.max_checks,
            interval=args.check_interval
        )
        product.setdefault('status_history', [])
        product['status_history'].extend(status_history)

        if not granules:
            product['error'] = "Product not ready or failed"
            updated_products.append(product)
            continue

        
        downloaded_files = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:  # adjust workers
            futures = {
                executor.submit(
                    download_file,
                    granule['uri'],
                    dest_path_folder / os.path.basename(urlparse(granule['uri']).path),
                    session,
                    # auth.token['access_token'],
                    args.max_retries,
                    args.retry_delay,
                    args.timeout
                ): granule for granule in granules
            }
            for future in as_completed(futures):
                granule = futures[future]
                dest_path_folder = save_data_loc.joinpath('Downloaded_Data')
                os.makedirs(dest_path_folder, exist_ok=True)
                dest_path = dest_path_folder / os.path.basename(urlparse(granule['uri']).path)
                if future.result():
                    downloaded_files.append(str(dest_path))

        if downloaded_files:
            product.update({
                'downloaded': True,
                'downloaded_at': time.strftime("%Y-%m-%dT%H:%M:%S"),
                'downloaded_files': downloaded_files,
                'final_state': 'SUCCESS'
            })
            successful_downloads += 1
        else:
            product['final_state'] = 'DOWNLOAD_FAILED'

        updated_products.append(product)

    # Save JSON once instead of every loop
    with open(args.input_file, 'w') as f:
        json.dump(updated_products, f, indent=2)

    print(f"\nDownload complete. {successful_downloads}/{len(products)} products downloaded")



def parse_arguments():
    parser = argparse.ArgumentParser(description="Download SWODLR requested products")
    parser.add_argument(
        "--save_data_loc",
        type=str,
        required=True,
        help="Directory to save downloaded products"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="netrc",
        help="Authentication strategy (default: netrc)"
    )
    parser.add_argument(
        "--max_checks",
        type=int,
        default=30,
        help="Maximum status check attempts (default: 30)"
    )
    parser.add_argument(
        "--check_interval",
        type=int,
        default=60,
        help="Seconds between status checks (default: 60)"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum download retries (default: 3)"
    )
    parser.add_argument(
        "--retry_delay",
        type=int,
        default=30,
        help="Seconds to wait before retrying a failed download (default: 10)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Download timeout in seconds (default: 300)"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="SWODLR_generated_products.json",
        help="JSON file with product list (default: SWODLR_generated_products.json)"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Number of parallel download workers (default: CPU cores - 1)"
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default="https://swodlr.podaac.earthdatacloud.nasa.gov/api/graphql",
        help="SWODLR GraphQL API URL (default: official API URL - https://swodlr.podaac.earthdatacloud.nasa.gov/api/graphql)"
    )
    return parser.parse_args()




if __name__ == "__main__":
    args = parse_arguments()
    try:
        download_products(args)
    except Exception as e:
        print(f"Fatal error: {e}")