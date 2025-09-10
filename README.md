# Utilizing SWOT SWODLR Service for Decision-Ready Information on Surface Water Management

## Overview

This project demonstrates how to leverage NASA’s Surface Water and Ocean Topography (SWOT) mission and its SWODLR (SWOT On-Demand Level 2 Raster Generation) service to generate customized water surface elevation (WSE) rasters (with spatial resolution as low as 90m). The goal is to transform raw satellite data into decision-ready information that can support effective surface water management strategies.

## Features  

- **Customized SWOT Raster Generation via SWODLR**  
  - Accesses the **SWODLR API** to generate raster products with resolutions as fine as **90 m**.  
  - Since SWODLR is **GraphQL-based** (a language unfamiliar to many users), this framework leverages **Python** to simplify access and lower the barrier of entry.  
  - Inputs are simplified: users only need to provide a **start/end date**, **region of interest** (shapefile, GeoJSON, or geographic bounds), and the desired **raster resolution**.  

- **Flood Propagation Analysis**  
  - Includes a **dominant patch module** to study the spread and dynamics of floods.  

    - Records the history of water clusters across timestamps to analyze:  
        - **Expansion** or **shrinkage** of water blobs  
        - **Merging** of multiple water bodies  
        - **Fragmentation** of larger blobs into smaller ones  

- **Visualization Tools**  
  - Generates quick **GIFs** for individual SWOT granules and flood propagation outputs.  

> ⚠️ **Important:**  
> Users can also use this framework for their analysis with **standard SWOT rasters** via the EarthData download component.  
> However, these are fixed at **100 m resolution**. To generate rasters with **customized resolutions**, the **SWODLR component** must be used.  

## Case Studies

The framework has been tested on selected water bodies, particularly in flood-prone contexts, demonstrating:
- The added value of customized rasters compared to standard products.
- Potential applications for decision-ready insights in water management.

## Limitations

SWODLR supports limited data collection (PGC0) 

## Getting Started
### Prerequisites
- Python 3.9+
- Dependencies listed in environment.yaml

### Installation
```
mkdir <project-dir>
cd <project-dir>
git clone https://github.com/shahzaib1007/SWOT_SWODLR_Python.git  
pip install -r requirements.txt
```

### Usage

The framework supports two modes of operation:  
1. **Customized SWOT Rasters via SWODLR** (flexible resolution, down to 90 m)  
2. **Standard SWOT Rasters via EarthData** (fixed at 100 m resolution)  

#### 1. Generating Customized SWOT Rasters - SWODLR
**Step 1: Request raster products**
```
python SWODLR_Python_Request.py --shapefile <path-to-your-shapefile> --output_file <path-where-you-want-to-download-swodlr-data>/SWODLR_generated_products.json --start_date <interested-date> --end_date <interested-date> --raster_resolution 90
```

**Step 2: Download raster data**
```
python SWODLR_Python_Download --save_data_loc <path-where-you-want-to-download-swodlr-data> --input_file <path-where-you-want-to-download-swodlr-data>/SWODLR_generated_products.json
```

**Step 3: Generate raster time series (GIF)**
```
python SWODLR_Python_Raster_TS --save_data_loc <path-where-you-downloaded-swodlr-data> --shapefile <path-to-your-shapefile>
```

**Step 4: Analyze flood propagation with dominant patch module**
```
python dominant_patch.py --save_data_loc <path-where-you-downloaded-swodlr-data> --overlay <path-to-your-shapefile> --store_frame
```


#### 2. Utilizing Standard SWOT Raster - EarthData
**Step 1: Download raster products**
```
python EarthData_Automated_Download.py --shapefile <path-to-your-shapefile> --save_data_loc <path-where-you-want-to-download-earthdata-data> --start_date <interested-date> --end_date <interested-date> 
```

**Step 2: Generate raster time series**
```
python EarthData_Automated_Raster_TS.py --overlay <path-to-your-shapefile> --save_data_loc <path-where-you-downloaded-earthdata-data> 
```

**Step 3: Analyze flood propagation with dominant patch module**
```
python dominant_patch.py --save_data_loc <path-where-you-downloaded-earthdata-data> --overlay <path-to-your-shapefile> --store_frame
```


> **Tip:**
>  - Use the SWODLR workflow if you need customized raster resolutions (e.g., 90 m).  
>  - Use the EarthData workflow if standard 100 m rasters are sufficient.  
>  - You can view all available options for any script by appending --help or -h. For e.g. 
    ```
    python SWODLR_Python_Request.py --help
    ```

