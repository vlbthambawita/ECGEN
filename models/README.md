# Models

ECG generation model architectures organized by category.

## Structure

- `vae/` - Variational Autoencoder models
- `gan/` - Generative Adversarial Network models
- `diffusion/` - Diffusion models (future)
- `ssm/` - State Space Models (future)
- `components/` - Shared components (encoders, decoders, blocks)

## Adding New Models

1. Choose the appropriate category folder
2. Create your model file
3. Add imports to the category's `__init__.py`
4. Update the category's README.md
