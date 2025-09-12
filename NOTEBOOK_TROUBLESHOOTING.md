# Inference_decoders.ipynb - Bug Fixes and Setup Guide

## Purpose of the Notebook

The `Inference_decoders.ipynb` notebook demonstrates how to use pretrained lightweight decoders with the Aurora weather foundation model to predict hydrological variables that are not present in the original Aurora model. Specifically, it predicts:

- **Total precipitation (MSWEP format)** - 6-hour accumulated precipitation from MSWEP dataset
- **Potential evaporation** - 6-hour accumulated from ERA5
- **Runoff** - instantaneous from ERA5  
- **Soil water content** - weighted sum of volumetric soil water layers down to 1 meter

The notebook follows this workflow:
1. **Data Download**: Automatically downloads ERA5 reanalysis data via Climate Data Store API
2. **Model Loading**: Loads the Aurora foundation model and pretrained hydrological decoders
3. **Inference**: Makes predictions using Aurora's latent representations as input to the decoders
4. **Visualization**: Compares decoder predictions with ERA5 reference data

## Bug Fixes Applied

### 1. Missing Python Module (`aurora.model.aurora_lite`)

**Problem**: `ModuleNotFoundError: No module named 'aurora.model.aurora_lite'`

**Root Cause**: 
- Missing `__init__.py` file in the `aurora` directory
- Notebook running from `/etc/python` instead of project directory

**Solution Applied**:
```python
# 1. Created missing __init__.py file
touch /scratch/qhuang62/fanny-hydro-decoder/aurora/__init__.py

# 2. Add Python path setup at the beginning of notebook:
import sys
import os

project_dir = '/scratch/qhuang62/fanny-hydro-decoder'
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
os.chdir(project_dir)
```

### 2. Flash Attention Dependency Issue

**Problem**: `ModuleNotFoundError: No module named 'flash_attn'`

**Root Cause**: The `flash_attn` package requires CUDA compilation tools which aren't available in the environment.

**Solution Applied**: Made `flash_attn` import optional in `aurora/model/swin3d.py`:
```python
# Before (line 25)
from flash_attn import flash_attn_qkvpacked_func

# After (lines 25-30)
try:
    from flash_attn import flash_attn_qkvpacked_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    flash_attn_qkvpacked_func = None
```

Added fallback to vanilla attention when flash attention is unavailable:
```python
# Modified attention mechanism (lines 171-180)
if HAS_FLASH_ATTN:
    ### use flash-attn ###
    qkv = rearrange(qkv, "qkv B H N D -> B N qkv H D")
    x = flash_attn_qkvpacked_func(qkv, dropout_p=attn_dropout)
    x = rearrange(x, "B N H D -> B H N D")
else:
    ### use vanilla attn ###
    x = F.scaled_dot_product_attention(q, k, v, dropout_p=attn_dropout)
```

### 3. Corrupted NetCDF Data File

**Problem**: `OSError: [Errno -51] NetCDF: Unknown file format`

**Root Cause**: The hydrological data file `2020-01-01-hydrological.nc` was downloaded as a ZIP archive instead of NetCDF format.

**Solution Applied**:
```bash
# 1. Extracted the ZIP file
cd /scratch/qhuang62/fanny-hydro-decoder/data/downloads
unzip -o 2020-01-01-hydrological.nc

# 2. Combined extracted files into proper NetCDF format
python -c "
import xarray as xr
instant = xr.open_dataset('data_stream-oper_stepType-instant.nc')  # swvl1, swvl2, swvl3
accum = xr.open_dataset('data_stream-oper_stepType-accum.nc')      # pev, ro
combined = xr.merge([instant, accum])
combined.to_netcdf('2020-01-01-hydrological.nc')
"

# 3. Cleaned up temporary files
rm data_stream-oper_stepType-*.nc
```

### 4. Variable Naming Mismatch

**Problem**: `KeyError: 'swvl_1'`

**Root Cause**: The notebook code used variable names with underscores (`swvl_1`, `swvl_2`, `swvl_3`) but the actual ERA5 dataset contains variables without underscores (`swvl1`, `swvl2`, `swvl3`).

**Solution Applied**: Updated variable names in the soil water content calculation:
```python
# Before
target = 0.07*hydro_vars_ds["swvl_1"].sel(valid_time=preds_org.metadata.time[0]).values + \
            0.21*hydro_vars_ds["swvl_2"].sel(valid_time=preds_org.metadata.time[0]).values + \
            0.72*hydro_vars_ds["swvl_3"].sel(valid_time=preds_org.metadata.time[0]).values

# After  
target = 0.07*hydro_vars_ds["swvl1"].sel(valid_time=preds_org.metadata.time[0]).values + \
            0.21*hydro_vars_ds["swvl2"].sel(valid_time=preds_org.metadata.time[0]).values + \
            0.72*hydro_vars_ds["swvl3"].sel(valid_time=preds_org.metadata.time[0]).values
```

## Setup Requirements

1. **Climate Data Store Account**: Sign up at https://cds.climate.copernicus.eu/
2. **Install Dependencies**:
   ```bash
   pip install cdsapi xarray netcdf4 torch matplotlib
   ```
3. **Configure CDS API**: Follow cdsapi setup instructions for your API key

## Running After Kernel Restart

If you restart the notebook kernel, make sure to run this setup cell first:

```python
import sys
import os

# Add project directory to Python path
project_dir = '/scratch/qhuang62/fanny-hydro-decoder'
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

os.chdir(project_dir)
print(f"Working directory: {os.getcwd()}")
print(f"Project dir in path: {project_dir in sys.path}")
```

The data files will persist and won't need to be re-downloaded due to the notebook's existence checks.