# Diffusion Models

Diffusion and flow matching live in the `ecgen` package:

- `ecgen.families.continuous` — `DiffusionModule` (`diffusion`), `FlowMatchingModule` (`flow_matching`),
  `LatentDiffusionModule` (`latent_diffusion`), `LatentFlowMatchingModule` (`latent_flow_matching`)
- `ecgen.processes` — `ddpm`, `rectified_flow`
- `ecgen.nn.backbones` — `unet1d`, `dit1d`, `ssm`, `mamba`

Configs: `configs/experiments/diffusion_*.yaml`, `flow_matching_mimic.yaml`,
`latent_diffusion_mimic.yaml`. Docs: docs/models/diffusion.md.
