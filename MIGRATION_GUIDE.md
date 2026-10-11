# Migration Guide

## 2026-10: training moved into the `ecgen` package (Phase 0)

Training is now part of the published package and runs through one runner:
`ecgen train configs/experiments/<x>.yaml [key=value ...]` (see `PLAN_TRAINING_LIBRARY.md`).
The old import paths below still work for one release (they re-export the new modules).

| Old | New |
|---|---|
| `models.vae.*` | `ecgen.families.vae` |
| `models.vqvae.*` | `ecgen.families.vqvae` |
| `models.gan.pulse2pulse` / `ecgen.models.pulse2pulse` | `ecgen.families.gan.pulse2pulse` |
| `ecgen.models.vae` | `ecgen.families.vae` (+ `ecgen.nn` for the blocks) |
| `models.components.{blocks,encoders,decoders}` | `ecgen.nn.{blocks,encoders,decoders}` |
| `training.losses.{vae,vqvae,gan}_losses` | `ecgen.families.{vae,vqvae,gan}.losses` |
| `training.callbacks.*` / `ecgen.training.callbacks` | `ecgen.training.callbacks.*` |
| `utils.{config,io,logging,metadata,seed}` | `ecgen.utils.*` |
| `data.datasets.base.BaseECGDataset` | `ecgen.data.datasets.BaseECGDataset` (samples use `"signal"`, not `"ecg_signals"`) |
| `data.datasets.mimic.MIMICIVECGDataset` (tuples) | `ecgen.data.datasets.MIMICIVECG` (dicts) + `ecgen.data.datamodule.ECGDataModule` |
| `scripts/train/train_{vae_mimic,vqvae_mimic,prior_mimic,pulse2pulse}.py` | `ecgen train <config>` (scripts removed) |
| `training/trainers/gan_trainer.py`, `python -m ecgen.training.train` | `ecgen train <config>` (old entry points forward to it) |
| `--vqvae-checkpoint <ckpt>` | `model.params.vqvae_checkpoint=<ckpt>` |

Config files changed shape: `model: {name, params}` (flat), `data: {dataset: {name, params: {root, ...}}, batch_size, ...}`,
`checkpoint:`, and a `callbacks:` list that is now honoured (the `visualization:` section became a
`vae_visualization` callback). Pulse2Pulse now trains on leads I, II, V1–V6 (`leads: independent8`).

---

## Earlier reorganization

Guide for migrating from the old structure to the new reorganized structure.

## What Changed

The repository has been reorganized with a clear, logical folder structure:

### Old Structure
```
ECGEN/
└── src/ecgen/
    ├── data/
    ├── models/
    ├── training/
    ├── evaluation/
    └── utils/
```

### New Structure
```
ECGEN/
├── data/datasets/
├── models/{vae,gan,diffusion,ssm}/
├── training/{callbacks,losses,metrics,trainers}/
├── evaluation/metrics/
├── visualization/
├── utils/
├── scripts/{train,generate,evaluate,data,shell}/
├── configs/{dataset,model,training,experiments}/
└── docs/
```

## Import Changes

### Models

**Old:**
```python
from ecgen.models.vae import VAE1D, VAELightning, VAEConfig
```

**New:**
```python
from models.vae import VAE1D, VAELightning, VAEConfig
```

### Data

**Old:**
```python
from ecgen.data.mimic_dataset import MIMICIVECGDataset
```

**New:**
```python
from data.datasets.mimic.dataset import MIMICIVECGDataset
```

### Training

**Old:**
```python
from ecgen.training.callbacks import VAEVisualizationCallback
from ecgen.training.losses import vae_loss
```

**New:**
```python
from training.callbacks.visualization import VAEVisualizationCallback
from training.losses.vae_losses import vae_loss
```

### Utils

**Old:**
```python
from ecgen.utils.seed import set_global_seed
```

**New:**
```python
from utils.seed import set_global_seed
```

## Path Updates

### Scripts

Training scripts moved from `scripts/` to `scripts/train/`:
- `scripts/train_vae_mimic.py` → `scripts/train/train_vae_mimic.py`

Shell scripts moved to `scripts/shell/`:
- `scripts/run_*.sh` → `scripts/shell/run_*.sh`

### Configs

Trainer configs renamed to training:
- `configs/trainer/` → `configs/training/`

Model configs organized by category:
- `configs/model/` → `configs/model/{vae,gan,diffusion,ssm}/`

### Documentation

All documentation consolidated in `docs/`:
- Root `.md` files → `docs/{getting_started,models,training,etc.}/`

## Migration Steps

### For Existing Scripts

1. Update imports to use new paths
2. Update `sys.path` to point to project root instead of `src/`
3. Update config file paths

### For Existing Configs

1. Update paths in YAML files
2. Move configs to appropriate subdirectories

### For Custom Code

1. Update all `ecgen.*` imports to new structure
2. Test imports: `python -c "from models.vae import VAE1D"`
3. Run tests to verify changes

## Backward Compatibility

The old `src/ecgen/` structure is temporarily kept for backward compatibility.

To use the new structure:
1. Update your imports as shown above
2. Update your `sys.path` to include project root
3. Test thoroughly

## Benefits of New Structure

1. **Clearer Organization**: Logical grouping by function
2. **Scalability**: Easy to add new model categories and datasets
3. **Better Documentation**: Organized docs structure
4. **Improved Navigation**: Intuitive folder hierarchy
5. **Future-Ready**: Prepared for diffusion and SSM models

## Getting Help

- See [docs/README.md](docs/README.md) for documentation index
- Check [PROJECT_STRUCTURE.md](docs/project_structure.md) for detailed structure
- Review examples in `scripts/train/` for updated patterns

## Timeline

- **Phase 1 (Current)**: New structure created, old structure maintained
- **Phase 2 (Next)**: Update all scripts and tests
- **Phase 3 (Future)**: Remove old `src/ecgen/` structure
- **Phase 4 (Final)**: Complete migration

## Questions?

Open an issue or check the documentation in `docs/`.
