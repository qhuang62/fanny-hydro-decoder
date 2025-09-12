# Fanny Hydro Decoder: Technical Documentation

## Overview

This repository implements lightweight MLP-based decoders for the Aurora weather foundation model to predict hydrological variables not present in the original model. The approach leverages Aurora's rich latent representations from its Perceiver-based encoder to predict new physical processes without retraining the entire foundation model.

## Architecture Overview

### Aurora Foundation Model Pipeline
```
Atmospheric Data → Encoder → Latent Space → Original Decoder → Weather Variables
                             ↓
                    New MLP Decoders → Hydrological Variables
```

The system works by:
1. **Aurora Encoder**: Processes multi-modal weather data into a unified latent representation
2. **Latent Extraction**: Captures the rich 1024-dimensional embeddings from Aurora's encoder
3. **MLP Decoders**: Lightweight neural networks that map latent features to hydrological variables

## Core Components

### 1. AuroraLite Model (`aurora/model/aurora_lite.py`)

The `AuroraLite` class extends the original Aurora model to expose latent representations:

```python
class AuroraLite(torch.nn.Module):
    def forward(self, batch: Batch) -> tuple[Batch, torch.Tensor]:
        # Standard Aurora forward pass
        preds_org = super().forward(batch)
        
        # Extract latent representations for decoders
        latent_decoder = self.encoder.latent  # Shape: (B, L, D)
        
        return preds_org, latent_decoder
```

**Key Modification**: The only change from the original Aurora is returning both the standard predictions and the latent representations from the encoder.

### 2. MLP Decoder Architecture (`aurora/model/decoder_lite.py`)

#### MLPDecoderLite Class

The core decoder uses simple Multi-Layer Perceptrons to map from latent space to physical variables:

```python
class MLPDecoderLite(nn.Module):
    def __init__(self, surf_vars_new, patch_size=4, embed_dim=1024, hidden_dims=[512, 512, 256]):
        super().__init__()
        
        # Separate MLP head for each hydrological variable
        self.surf_heads = nn.ParameterDict({
            name: MLP(embed_dim, hidden_dims + [patch_size**2]) 
            for name in surf_vars_new
        })
```

#### Architecture Details

**Input**: Latent representations from Aurora encoder
- Shape: `(B, L, D)` where B=batch, L=spatial locations, D=1024 (embed_dim)

**Processing Pipeline**:
1. **Variable-Specific MLPs**: Each hydrological variable has its own MLP head
2. **Patch-Based Output**: Each MLP outputs `patch_size²` values (16 for 4×4 patches)
3. **Unpatchify**: Reshape patch outputs back to spatial grid
4. **Denormalization**: Apply inverse transformations to get physical units

```python
def forward(self, latent_decoder, lat, lon):
    # Apply variable-specific MLPs to latent features
    x_surf = torch.stack([
        self.surf_heads[name](latent_decoder[..., :1, :]) 
        for name in self.surf_vars_new
    ], dim=-1)
    
    # Reshape and unpatchify to spatial grid
    surf_preds = unpatchify(x_surf, len(self.surf_vars_new), H, W, self.patch_size)
    
    # Create variable dictionary
    pred_new = {v: surf_preds[:, i] for i, v in enumerate(self.surf_vars_new)}
    
    # Denormalize to physical units
    pred_new = {k: unnormalise_surf_var(v, k, stats=self.stats) 
                for k, v in pred_new.items()}
```

## Latent Space Analysis

### Encoder Architecture

Aurora's encoder uses a **Perceiver3D** architecture that creates rich multi-modal embeddings:

```python
# From aurora/model/encoder.py
class Perceiver3DEncoder(nn.Module):
    def __init__(self, surf_vars, static_vars, atmos_vars, embed_dim=1024, ...):
        # Multi-modal patch embedding
        self.surf_embed = VariablePatchEmbed(surf_vars, patch_size, embed_dim)
        self.atmos_embed = LevelPatchEmbed(atmos_vars, patch_size, embed_dim, levels)
        
        # Perceiver resampler for cross-attention
        self.perceiver = PerceiverResampler(...)
```

### Patch-Based Representation

**Spatial Patching**: The encoder divides the global grid into patches (typically 4×4):
- **Input Resolution**: 720×1440 (0.25° global grid)  
- **Patch Size**: 4×4 pixels
- **Patches**: 180×360 = 64,800 patches
- **Latent Dimensions**: 1024 per patch

**Multi-Modal Fusion**: Each patch embedding combines:
- **Surface variables**: 2m temperature, winds, pressure
- **Atmospheric variables**: Temperature, humidity, geopotential at 13 pressure levels
- **Static variables**: Topography, land-sea mask, soil type
- **Temporal encoding**: Lead time, absolute time
- **Positional encoding**: Latitude, longitude

### Latent Space Properties

The 1024-dimensional latent vectors encode:
1. **Local Atmospheric State**: Temperature, humidity, pressure profiles
2. **Dynamic Processes**: Wind patterns, vertical motion
3. **Surface Conditions**: Land-ocean contrasts, topographic effects
4. **Temporal Context**: Seasonal cycles, time-of-day effects

## Data Transformations (`transform_data.py`)

### Variable-Specific Preprocessing

Different physical variables require specific transformations for stable training:

```python
def transform_data(data, var_name, eps=1e-5, direct=True):
    if var_name in ["tp", "tp_mswep", "r"]:  # Precipitation, runoff
        if direct:
            return np.log(1 + data/eps)  # Log transform for skewed distributions
        else:
            return eps*(np.exp(data) - 1)  # Inverse transform
            
    elif var_name in ["pe", "e"]:  # Evaporation
        if direct:
            return -5e3*data  # Scaling for numerical stability
        else:
            return data/(-5e3)
            
    # Other variables use identity transform
    return data
```

**Rationale**:
- **Precipitation/Runoff**: Log transformation handles extreme values and zero-inflated distributions
- **Evaporation**: Scaling prevents gradient issues with small values
- **Soil Water**: Direct prediction as values are well-behaved

## Training Strategy

### Lightweight Fine-tuning Approach

1. **Frozen Foundation Model**: Aurora encoder and backbone remain unchanged
2. **Trainable Decoders**: Only the MLP heads are trained (< 1M parameters vs 1.3B for full Aurora)
3. **Multi-Task Learning**: All hydrological variables trained jointly
4. **Reference Datasets**:
   - **Precipitation**: MSWEP (6-hour accumulated, log-transformed)
   - **Potential Evaporation**: ERA5 (6-hour accumulated)  
   - **Runoff**: ERA5 (instantaneous)
   - **Soil Water Content**: ERA5 (weighted sum of top 3 layers)

### Loss Function

```python
# Multi-task loss combining all variables
loss = sum(mse_loss(pred[var], target[var]) for var in surf_vars_new)
```

## Repository Structure

```
├── Inference_decoders.ipynb          # Main demonstration notebook
├── lite-decoder.ckpt                 # Pretrained decoder weights
├── transform_data.py                 # Data transformation utilities
├── aurora/                           # Modified Aurora components
│   ├── __init__.py                   # Package initialization
│   ├── batch.py                      # Data batch handling
│   ├── area.py                       # Spherical area calculations
│   ├── normalisation.py              # Data normalization
│   ├── rollout.py                    # Multi-step prediction
│   └── model/                        # Neural network components
│       ├── __init__.py               # Model package init
│       ├── aurora_lite.py            # Extended Aurora with latent output
│       ├── decoder_lite.py           # MLP decoders for hydrology
│       ├── decoder.py                # Original Aurora decoder
│       ├── encoder.py                # Perceiver3D encoder
│       ├── swin3d.py                 # Swin Transformer backbone
│       ├── perceiver.py              # Cross-attention components  
│       ├── patchembed.py             # Patch embedding layers
│       ├── posencoding.py            # Positional encodings
│       ├── fourier.py                # Fourier feature expansions
│       ├── film.py                   # Feature modulation layers
│       ├── lora.py                   # Low-rank adaptation
│       └── util.py                   # Utility functions
├── data/downloads/                   # ERA5 data storage
│   ├── static.nc                     # Time-invariant variables
│   ├── 2020-01-01-surface-level.nc  # Surface meteorology
│   ├── 2020-01-01-atmospheric.nc    # 3D atmospheric data
│   └── 2020-01-01-hydrological.nc   # Hydrological variables
└── aurora_decoders.png               # Architecture illustration
```

## Key Implementation Details

### Patch Reconstruction

The `unpatchify` function reconstructs spatial fields from patch-based predictions:

```python
def unpatchify(x: torch.Tensor, V: int, H: int, W: int, P: int) -> torch.Tensor:
    """
    Args:
        x: Patch predictions (B, L, V*P*P)
        V: Number of variables
        H, W: Spatial dimensions
        P: Patch size
    Returns:
        Reconstructed spatial fields (B, V, H, W)
    """
    B, L = x.shape[:2]
    
    # Reshape patch predictions
    x = rearrange(x, "B (H W) (V P1 P2) -> B V (H P1) (W P2)", 
                  H=H//P, W=W//P, P1=P, P2=P, V=V)
    
    return x[:, :, :H, :W]  # Crop to exact dimensions
```

### Multi-Scale Processing

The decoders operate at Aurora's native patch resolution then reconstruct full-resolution outputs:

1. **Encoder Resolution**: 180×360 patches (4× downsampling)
2. **Latent Processing**: MLP operations at patch level  
3. **Decoder Resolution**: 720×1440 full grid (unpatchified)

This design balances computational efficiency with spatial detail preservation.

## Performance Characteristics

- **Model Size**: ~500K parameters (decoders only) vs 1.3B (full Aurora)
- **Training Time**: Hours vs weeks for full model training
- **Inference Speed**: Minimal overhead over standard Aurora
- **Memory Usage**: Modest increase due to latent storage

The approach demonstrates how foundation model embeddings can be efficiently leveraged for new tasks without extensive retraining.