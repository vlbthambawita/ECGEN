# State Space Models

State-space backbones live in the `ecgen` package:

- `ecgen.nn.backbones.ssm.S4D` — diagonal state-space layer (S4D), pure PyTorch, FFT convolution
- `ecgen.nn.backbones.ssm.SSMBackbone` — SSSD-style residual stack of bidirectional S4D layers (registry name `ssm`)
- `ecgen.nn.backbones.ssm.MambaBackbone` — the same stack with bidirectional Mamba layers (`mamba`, needs `mamba-ssm`)

Use them as the `backbone` of the `diffusion`, `flow_matching` or latent families,
e.g. `configs/experiments/diffusion_mimic_ssm.yaml`. See docs/models/diffusion.md.
