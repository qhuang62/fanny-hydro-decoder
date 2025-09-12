# Repository Structure and File Documentation

## Repository Overview

This repository implements lightweight decoders for the Aurora weather foundation model to predict hydrological variables. The codebase extends Aurora's architecture with minimal modifications while adding new capabilities for precipitation, evaporation, runoff, and soil water content prediction.

## Root Directory Files

### Core Files

| File | Purpose | Key Contents |
|------|---------|--------------|
| `Inference_decoders.ipynb` | Main demonstration notebook | Complete pipeline from data download to prediction visualization |
| `lite-decoder.ckpt` | Pretrained model weights | Trained MLP decoder parameters for all hydrological variables |
| `transform_data.py` | Data preprocessing utilities | Variable-specific transformations (log, scaling) |
| `README.md` | Project overview | Installation, usage, citation information |
| `LICENSE` | MIT license | Copyright and usage terms |
| `aurora_decoders.png` | Architecture illustration | Visual diagram of Aurora + decoders system |

### Documentation Files (Generated)

| File | Purpose | Contents |
|------|---------|----------|
| `NOTEBOOK_TROUBLESHOOTING.md` | Setup and debugging guide | Bug fixes, environment setup, common issues |
| `TECHNICAL_DOCUMENTATION.md` | Deep technical analysis | Architecture details, latent space analysis, implementation |
| `REPOSITORY_STRUCTURE.md` | This file | Complete file and directory documentation |

## Aurora Package Structure

### Main Aurora Directory (`aurora/`)

| File | Purpose | Key Components |
|------|---------|----------------|
| `__init__.py` | Package initialization | Makes aurora importable as Python module |
| `batch.py` | Data batch handling | `Batch` class for meteorological data containers |
| `area.py` | Spherical geometry | Area calculations for global grids |
| `normalisation.py` | Data normalization | Statistical normalization for all variables |
| `rollout.py` | Multi-step prediction | Autoregressive forecasting capabilities |

### Model Components (`aurora/model/`)

#### Core Architecture Files

| File | Purpose | Key Classes | Description |
|------|---------|-------------|-------------|
| `__init__.py` | Model package init | - | Package initialization for model components |
| `aurora_lite.py` | **Modified Aurora model** | `AuroraLite` | Extended Aurora that outputs latent representations |
| `decoder_lite.py` | **New MLP decoders** | `MLPDecoderLite`, `Perceiver3DDecoderLite` | Lightweight decoders for hydrological variables |

#### Original Aurora Components

| File | Purpose | Key Classes | Technical Details |
|------|---------|-------------|-------------------|
| `encoder.py` | Perceiver3D encoder | `Perceiver3DEncoder` | Multi-modal patch embedding, cross-attention |
| `decoder.py` | Original Aurora decoder | `Perceiver3DDecoder` | Standard weather variable prediction |
| `swin3d.py` | Transformer backbone | `Swin3DTransformerBackbone`, `BasicLayer3D` | 3D Swin Transformer with window attention |

#### Supporting Architecture

| File | Purpose | Key Components | Function |
|------|---------|----------------|----------|
| `perceiver.py` | Cross-attention modules | `PerceiverResampler`, `MLP` | Latent-query cross-attention mechanism |
| `patchembed.py` | Patch embedding layers | `VariablePatchEmbed`, `LevelPatchEmbed` | Convert raw data to patch tokens |
| `posencoding.py` | Positional encodings | `pos_scale_enc` | Spatial and temporal position encoding |
| `fourier.py` | Feature expansions | `levels_expansion`, `variables_expansion` | Fourier-based feature engineering |
| `film.py` | Feature modulation | `AdaptiveLayerNorm` | Conditional normalization layers |
| `lora.py` | Low-rank adaptation | `LoRAMode`, `LoRARollout` | Parameter-efficient fine-tuning |
| `util.py` | Utility functions | `unpatchify`, `init_weights` | Common operations and initialization |

## Data Directory (`data/downloads/`)

### ERA5 Reanalysis Data

| File | Content | Variables | Temporal Resolution |
|------|---------|-----------|-------------------|
| `static.nc` | Time-invariant fields | `z` (geopotential), `lsm` (land-sea mask), `slt` (soil type) | Single time point |
| `2020-01-01-surface-level.nc` | Surface meteorology | `t2m`, `u10`, `v10`, `msl` | 6-hourly (00, 06, 12, 18 UTC) |
| `2020-01-01-atmospheric.nc` | 3D atmospheric data | `t`, `u`, `v`, `q`, `z` at 13 pressure levels | 6-hourly |
| `2020-01-01-hydrological.nc` | Hydrological variables | `pev`, `ro`, `swvl1`, `swvl2`, `swvl3` | Hourly (pev, ro), 6-hourly (swvl) |

### Data Specifications

- **Spatial Resolution**: 0.25° global grid (720×1440)
- **Spatial Coverage**: Global (-90° to 90° latitude, -180° to 180° longitude)
- **Temporal Coverage**: Single day (2020-01-01) for demonstration
- **Format**: NetCDF4 (HDF5-based)

## File Dependencies and Relationships

### Import Hierarchy

```
Inference_decoders.ipynb
├── aurora.batch (Batch, Metadata)
├── aurora.model.aurora_lite (AuroraLite)
├── aurora.model.decoder_lite (MLPDecoderLite)
└── transform_data (data transformations)

AuroraLite
├── aurora.model.encoder (Perceiver3DEncoder)
├── aurora.model.decoder_lite (Perceiver3DDecoderLite)  
├── aurora.model.swin3d (Swin3DTransformerBackbone)
└── aurora.model.lora (LoRA components)

MLPDecoderLite
├── aurora.normalisation (unnormalise_surf_var)
├── aurora.model.util (unpatchify, init_weights)
└── torchvision.ops (MLP)
```

### Data Flow Pipeline

```
Raw ERA5 Data (NetCDF)
    ↓
aurora.batch.Batch (data container)
    ↓
aurora.model.encoder.Perceiver3DEncoder (multi-modal embedding)
    ↓
aurora.model.swin3d.Swin3DTransformerBackbone (spatiotemporal processing)
    ↓
Latent Representations (B×L×1024)
    ↓
aurora.model.decoder_lite.MLPDecoderLite (hydrological prediction)
    ↓
transform_data.transform_data (denormalization)
    ↓
Physical Variables (mm/hour, m³/m³, etc.)
```

## Key Modifications from Original Aurora

### 1. Aurora Lite Extension (`aurora_lite.py`)

**Changes Made**:
- Added latent output to `forward()` method
- Exposed encoder representations for decoder training
- Maintained full backward compatibility

**Original**:
```python
def forward(self, batch: Batch) -> Batch:
    return super().forward(batch)  # Only predictions
```

**Modified**:
```python
def forward(self, batch: Batch) -> tuple[Batch, torch.Tensor]:
    preds = super().forward(batch)
    latent = self.encoder.latent  # Extract latents
    return preds, latent
```

### 2. New Decoder Implementation (`decoder_lite.py`)

**Key Features**:
- Variable-specific MLP heads
- Patch-based spatial reconstruction  
- Integrated data denormalization
- Support for both MLP and Perceiver architectures

### 3. Flash Attention Compatibility (`swin3d.py`)

**Issue**: Flash attention requires CUDA compilation
**Solution**: Optional import with fallback to vanilla attention

```python
try:
    from flash_attn import flash_attn_qkvpacked_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
```

## Usage Patterns

### 1. Model Loading
```python
# Load Aurora foundation model
model_aurora = AuroraLite(...)
model_aurora.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")

# Load pretrained decoders
model_decoder = MLPDecoderLite(surf_vars_new=["tp_mswep", "pe", "r", "swc"])
checkpoint = torch.load("./lite-decoder.ckpt")
model_decoder.load_state_dict(checkpoint)
```

### 2. Inference Pipeline
```python
# Standard Aurora prediction + latent extraction
with torch.inference_mode():
    preds_org, latent = model_aurora.forward(batch)
    
    # Hydrological variable prediction
    preds_new = model_decoder.forward(latent, batch.metadata.lat, batch.metadata.lon)
    
    # Denormalize predictions
    preds_new = {k: transform_data(v, k, direct=False) for k, v in preds_new.items()}
```

### 3. Data Preprocessing
```python
# Variable-specific transformations
def transform_data(data, var_name, direct=True):
    if var_name in ["tp", "tp_mswep", "r"]:
        return np.log(1 + data/1e-5) if direct else 1e-5*(np.exp(data) - 1)
    elif var_name in ["pe", "e"]:  
        return -5e3*data if direct else data/(-5e3)
    return data
```

## Development Notes

### Adding New Variables

1. **Update `surf_vars_new`** in decoder initialization
2. **Add transformation** in `transform_data.py`
3. **Retrain decoder** with new target variable
4. **Update normalization** statistics if needed

### Model Architecture Extensions

The modular design allows easy extension:
- **New encoders**: Replace `Perceiver3DEncoder`
- **New backbones**: Replace `Swin3DTransformerBackbone`  
- **New decoders**: Add to `decoder_lite.py`
- **New variables**: Extend MLP heads

### Performance Optimization

- **Mixed precision**: Set `autocast=True` in AuroraLite
- **Gradient checkpointing**: Use in Swin3D layers
- **Batch processing**: Optimize for GPU memory usage
- **LoRA fine-tuning**: For parameter-efficient adaptation

This architecture balances flexibility, performance, and maintainability while enabling rapid development of new weather prediction capabilities.