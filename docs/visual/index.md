# Visual Guide

This section explains every implemented model family with diagrams: what goes in, what the network
does to the tensor, what comes out, and how the training loss is computed. Most diagrams are
animated or interactive. Hover over (or tap) a block to see the exact layer, and use the controls under
a diagram to change its inputs.

| Page | Model | Generates from |
|------|-------|----------------|
| [VAE](vae.md) | `VAE1D` / `VAELightning` | a Gaussian latent sequence `z ~ N(0, I)` |
| [VQ-VAE + Prior](vqvae.md) | `VQVAE1D` + `PixelCNNPrior` | code indices sampled by an autoregressive prior |
| [Pulse2Pulse GAN](pulse2pulse.md) | `WaveGANGenerator` + `WaveGANDiscriminator` | uniform noise with the shape of an ECG |
| [Diffusion (DDPM)](diffusion.md) | `DiffusionModule` + `DDPMProcess` + `UNet1d` | Gaussian noise `(n, 12, 5000)`, denoised in ~50 DDIM steps |
| [Flow Matching](flow_matching.md) | `FlowMatchingModule` + `RectifiedFlowProcess` + `UNet1d` | Gaussian noise, moved along a learned velocity field with Euler / Heun steps |

All shapes and hyper-parameters in the diagrams are the **defaults** in `src/ecgen/families/` and
`src/ecgen/nn/`. The written reference for each model is under *Models*
([VAE](../models/vae.md), [VQ-VAE](../models/vqvae.md), [Pulse2Pulse](../models/gan.md),
[Diffusion and flow matching](../models/diffusion.md)).

## Legend

<div class="arch-legend">
<span><svg viewBox="0 0 34 16"><rect x="1" y="2" width="32" height="12" rx="2" class="tensor" style="fill:var(--arch-data-fill);stroke:var(--arch-data)"/></svg>tensor / data</span>
<span><svg viewBox="0 0 34 16"><rect x="1" y="2" width="32" height="12" rx="2" style="fill:var(--arch-module-fill);stroke:var(--arch-module)"/></svg>trainable module</span>
<span><svg viewBox="0 0 34 16"><rect x="1" y="2" width="32" height="12" rx="2" style="fill:var(--arch-frozen-fill);stroke:var(--arch-frozen);stroke-dasharray:4 2"/></svg>frozen module</span>
<span><svg viewBox="0 0 34 16"><rect x="1" y="2" width="32" height="12" rx="2" style="fill:var(--arch-loss-fill);stroke:var(--arch-loss)"/></svg>loss term</span>
<span><svg viewBox="0 0 34 16"><line x1="1" y1="8" x2="33" y2="8" style="stroke:var(--arch-data);stroke-width:2;stroke-dasharray:6 4"/></svg>forward pass</span>
<span><svg viewBox="0 0 34 16"><line x1="1" y1="8" x2="33" y2="8" style="stroke:var(--arch-grad);stroke-width:2;stroke-dasharray:3 3"/></svg>gradient</span>
<span><svg viewBox="0 0 34 16"><line x1="1" y1="8" x2="33" y2="8" style="stroke:var(--arch-ok);stroke-width:2;stroke-dasharray:2 3"/></svg>skip connection</span>
<span><b style="font-family:var(--md-code-font-family)">sg[·]</b>&nbsp;stop-gradient (<code>.detach()</code>)</span>
</div>

## The input: a 12-lead ECG

Every model works on a tensor of shape `(batch, leads, samples)`. `MIMICIVECG` yields
12 leads × 5000 samples (10 s at 500 Hz) by default. Pulse2Pulse uses 8 channels
(`num_channels=8`, selected with `leads: independent8`: I, II, V1–V6).

<figure class="arch" data-widget="leads">
<svg viewBox="0 0 760 290" role="img" aria-label="Twelve-lead ECG grid">
<g data-lead-idx="0"><rect x="10" y="10" width="176" height="76" rx="3" class="plain"/><text x="16" y="24" class="sm">I</text><path data-ecg="14,26,168,56" data-lead="I" data-seed="1" data-beats="2.5" class="trace draw"/></g>
<g data-lead-idx="3"><rect x="196" y="10" width="176" height="76" rx="3" class="plain"/><text x="202" y="24" class="sm">aVR</text><path data-ecg="200,26,168,56" data-lead="aVR" data-seed="1" data-beats="2.5" class="trace draw"/></g>
<g data-lead-idx="6"><rect x="382" y="10" width="176" height="76" rx="3" class="plain"/><text x="388" y="24" class="sm">V1</text><path data-ecg="386,26,168,56" data-lead="V1" data-seed="1" data-beats="2.5" class="trace draw"/></g>
<g data-lead-idx="9"><rect x="568" y="10" width="176" height="76" rx="3" class="plain"/><text x="574" y="24" class="sm">V4</text><path data-ecg="572,26,168,56" data-lead="V4" data-seed="1" data-beats="2.5" class="trace draw"/></g>
<g data-lead-idx="1"><rect x="10" y="96" width="176" height="76" rx="3" class="plain"/><text x="16" y="110" class="sm">II</text><path data-ecg="14,112,168,56" data-lead="II" data-seed="1" data-beats="2.5" class="trace draw"/></g>
<g data-lead-idx="4"><rect x="196" y="96" width="176" height="76" rx="3" class="plain"/><text x="202" y="110" class="sm">aVL</text><path data-ecg="200,112,168,56" data-lead="aVL" data-seed="1" data-beats="2.5" class="trace draw"/></g>
<g data-lead-idx="7"><rect x="382" y="96" width="176" height="76" rx="3" class="plain"/><text x="388" y="110" class="sm">V2</text><path data-ecg="386,112,168,56" data-lead="V2" data-seed="1" data-beats="2.5" class="trace draw"/></g>
<g data-lead-idx="10"><rect x="568" y="96" width="176" height="76" rx="3" class="plain"/><text x="574" y="110" class="sm">V5</text><path data-ecg="572,112,168,56" data-lead="V5" data-seed="1" data-beats="2.5" class="trace draw"/></g>
<g data-lead-idx="2"><rect x="10" y="182" width="176" height="76" rx="3" class="plain"/><text x="16" y="196" class="sm">III</text><path data-ecg="14,198,168,56" data-lead="III" data-seed="1" data-beats="2.5" class="trace draw"/></g>
<g data-lead-idx="5"><rect x="196" y="182" width="176" height="76" rx="3" class="plain"/><text x="202" y="196" class="sm">aVF</text><path data-ecg="200,198,168,56" data-lead="aVF" data-seed="1" data-beats="2.5" class="trace draw"/></g>
<g data-lead-idx="8"><rect x="382" y="182" width="176" height="76" rx="3" class="plain"/><text x="388" y="196" class="sm">V3</text><path data-ecg="386,198,168,56" data-lead="V3" data-seed="1" data-beats="2.5" class="trace draw"/></g>
<g data-lead-idx="11"><rect x="568" y="182" width="176" height="76" rx="3" class="plain"/><text x="574" y="196" class="sm">V6</text><path data-ecg="572,198,168,56" data-lead="V6" data-seed="1" data-beats="2.5" class="trace draw"/></g>
<text x="380" y="282" class="sm c mono" data-role="shape">x : (B, 12, 5000)</text>
</svg>
<div class="arch-controls"><button type="button" data-leads="12">12 leads — VAE, VQ-VAE</button><button type="button" data-leads="8">8 leads — Pulse2Pulse</button></div>
<figcaption>Schematic traces (not real patient data), shown on the standard 3 × 4 layout. Channel order in the tensor is I, II, III, aVR, aVL, aVF, V1–V6.</figcaption>
</figure>

## How each family generates an ECG

Training differs a lot between the three families, but at sampling time each one is a fixed pipeline
from random numbers to a `(n, leads, 5000)` tensor:

<figure class="arch">
<svg viewBox="0 0 760 330" role="img" aria-label="Sampling pipelines of the three model families">
<text x="10" y="60" class="lg">VAE</text>
<g data-info="<b>VAELightning.sample(n, seq_length=5000)</b> draws z ~ N(0, I) of shape (n, 8, 625) and decodes it."><rect x="90" y="38" width="160" height="44" rx="6" class="tensor"/><text x="170" y="57" class="sm c">z ~ N(0, I)</text><text x="170" y="73" class="xs c mono muted">(n, 8, 625)</text></g>
<path d="M252,60 L408,60" class="edge flow"/>
<g data-info="<b>Decoder1D</b> — latent (8 ch, length 625) → 12 × 5000. See the shared backbone below."><rect x="410" y="38" width="84" height="44" rx="6" class="module"/><text x="452" y="64" class="sm c">Decoder1D</text></g>
<path d="M496,60 L518,60" class="edge flow"/>
<path data-ecg="522,34,218,52" data-lead="II" data-seed="3" class="trace draw"/>
<circle r="4" cy="60" class="packet"><animateMotion dur="2.4s" repeatCount="indefinite" path="M252,0 L520,0"/></circle>
<text x="10" y="160" class="lg">VQ-VAE</text>
<g data-info="<b>PixelCNNPrior.sample()</b> — draws 625 code indices one at a time, each from softmax(logits / temperature)."><rect x="90" y="138" width="110" height="44" rx="6" class="module"/><text x="145" y="157" class="sm c">PixelCNN prior</text><text x="145" y="173" class="xs c muted">autoregressive</text></g>
<path d="M202,160 L222,160" class="edge flow"/>
<g data-info="<b>Code indices</b> — (n, 625) integers in [0, 512)."><rect x="224" y="138" width="80" height="44" rx="6" class="tensor"/><text x="264" y="157" class="sm c">codes</text><text x="264" y="173" class="xs c mono muted">(n, 625)</text></g>
<path d="M306,160 L326,160" class="edge flow"/>
<g data-info="<b>Codebook lookup</b> — VectorQuantizer.get_codebook_entry: each index becomes its 64-d codebook vector → (n, 64, 625)."><rect x="328" y="138" width="70" height="44" rx="6" class="tensor"/><text x="363" y="157" class="sm c">codebook</text><text x="363" y="173" class="xs c mono muted">(n,64,625)</text></g>
<path d="M400,160 L408,160" class="edge flow"/>
<g data-info="<b>Decoder1D</b> of the frozen stage-1 VQ-VAE — 64 × 625 → 12 × 5000."><rect x="410" y="138" width="84" height="44" rx="6" class="module"/><text x="452" y="164" class="sm c">Decoder1D</text></g>
<path d="M496,160 L518,160" class="edge flow"/>
<path data-ecg="522,134,218,52" data-lead="II" data-seed="5" class="trace draw"/>
<circle r="4" cy="160" class="packet"><animateMotion dur="2.4s" repeatCount="indefinite" path="M202,0 L520,0"/></circle>
<text x="10" y="260" class="lg">GAN</text>
<g data-info="<b>Noise</b> — uniform in [−1, 1] with the full ECG shape (n, 8, 5000)."><rect x="90" y="238" width="160" height="44" rx="6" class="tensor"/><path data-ecg="96,242,148,24" data-kind="noise" data-seed="2" class="trace noise"/><text x="170" y="278" class="xs c mono muted">U(−1, 1) · (n, 8, 5000)</text></g>
<path d="M252,260 L288,260" class="edge flow"/>
<g data-info="<b>WaveGANGenerator</b> — a 1-D U-Net with kernel-25 convolutions and a tanh output."><rect x="290" y="238" width="204" height="44" rx="6" class="module"/><text x="392" y="257" class="sm c">WaveGANGenerator</text><text x="392" y="273" class="xs c muted">1-D U-Net, tanh</text></g>
<path d="M496,260 L518,260" class="edge flow"/>
<path data-ecg="522,234,218,52" data-lead="II" data-seed="8" data-kind="fake" class="trace fake draw"/>
<circle r="4" cy="260" class="packet"><animateMotion dur="2.4s" repeatCount="indefinite" path="M252,0 L520,0"/></circle>
</svg>
<div class="arch-info">Hover over a block for details.</div>
</figure>

## Shared backbone: `Encoder1D` / `Decoder1D`

The VAE and the VQ-VAE use the same convolutional encoder and decoder from `src/ecgen/nn/`.
They differ only in the latent bottleneck: the VAE samples a Gaussian latent with 8 channels, and the
VQ-VAE snaps a 64-channel latent to a codebook. Switch between them below:

<figure class="arch" data-widget="backbone" data-mode="vae">
<svg data-diagram="backbone" viewBox="0 0 760 278" role="img" aria-label="Encoder1D and Decoder1D block diagram"></svg>
<div class="arch-controls"><button type="button" data-mode="vae">VAE bottleneck (latent_channels = 8)</button><button type="button" data-mode="vqvae">VQ-VAE bottleneck (latent_channels = 64)</button></div>
<div class="arch-info">Hover over a block for the exact layer. Thin blocks are the stride-2 down/up-sampling convolutions.</div>
</figure>

Every coloured block in the stages is a `ResidualBlock1D`:

<figure class="arch">
<svg viewBox="0 0 760 130" role="img" aria-label="ResidualBlock1D">
<g data-info="Input h: (B, C_in, L)."><rect x="10" y="62" width="44" height="34" rx="4" class="tensor"/><text x="32" y="83" class="sm c">h</text></g>
<path d="M56,79 L82,79" class="edge flow"/>
<g data-info="Conv1d(C_in → C_out, k=3, padding=1)"><rect x="84" y="62" width="80" height="34" rx="4" class="module"/><text x="124" y="83" class="sm c">Conv k=3</text></g>
<path d="M166,79 L178,79" class="edge"/>
<g data-info="GroupNorm(8, C_out) — channel widths must be divisible by 8."><rect x="180" y="62" width="74" height="34" rx="4" class="module"/><text x="217" y="83" class="sm c">GN(8)</text></g>
<path d="M256,79 L268,79" class="edge"/>
<g data-info="SiLU activation: x·sigmoid(x)."><rect x="270" y="62" width="54" height="34" rx="4" class="module"/><text x="297" y="83" class="sm c">SiLU</text></g>
<path d="M326,79 L338,79" class="edge"/>
<g data-info="Conv1d(C_out → C_out, k=3, padding=1)"><rect x="340" y="62" width="80" height="34" rx="4" class="module"/><text x="380" y="83" class="sm c">Conv k=3</text></g>
<path d="M422,79 L434,79" class="edge"/>
<g data-info="GroupNorm(8, C_out)"><rect x="436" y="62" width="74" height="34" rx="4" class="module"/><text x="473" y="83" class="sm c">GN(8)</text></g>
<path d="M512,79 L524,79" class="edge"/>
<g data-info="SiLU activation (applied before the residual add)."><rect x="526" y="62" width="54" height="34" rx="4" class="module"/><text x="553" y="83" class="sm c">SiLU</text></g>
<path d="M582,79 L612,79" class="edge flow"/>
<circle cx="628" cy="79" r="14" class="ok"/><text x="628" y="84" class="lg c">+</text>
<path d="M644,79 L676,79" class="edge flow"/>
<g data-info="Output: (B, C_out, L) — length unchanged."><rect x="678" y="62" width="64" height="34" rx="4" class="tensor"/><text x="710" y="83" class="sm c">out</text></g>
<path d="M32,60 L32,24 L628,24 L628,63" class="edge skip"/>
<g data-info="<b>Shortcut</b> — identity when C_in = C_out, otherwise Conv1d(C_in → C_out, k=1)."><rect x="250" y="12" width="210" height="24" rx="12" class="ok"/><text x="355" y="28" class="xs c">shortcut: identity or 1×1 conv</text></g>
</svg>
<div class="arch-info">Hover over a layer.</div>
</figure>

### How long is the latent?

Only the first `len(channel_multipliers) − 1` stages downsample, each by 2 (`Conv1d(k=4, stride=2, padding=1)`),
and the decoder upsamples by the same factor. If the input length is not divisible by the factor the
reconstruction comes out shorter than the input:

<figure class="arch" data-widget="downsample">
<div class="arch-controls">
<label>stages <input type="range" name="stages" min="2" max="5" step="1" value="4"><output name="stages"></output></label>
<label>seq_length <select name="length"><option>2500</option><option selected>5000</option><option>4000</option><option>10000</option><option>1000</option><option>5003</option></select></label>
</div>
<div class="arch-info"></div>
</figure>

Use `VQVAELightning.downsample_factor` / `latent_length()` in code instead of hard-coding 8.

## At a glance

| | VAE | VQ-VAE + prior | Pulse2Pulse | Diffusion | Flow matching |
|---|---|---|---|---|---|
| Leads × samples | 12 × 5000 | 12 × 5000 | 8 × 5000 | 12 × 5000 | 12 × 5000 |
| Latent | continuous, `(8, 625)` | discrete, 625 indices into a 512 × 64 codebook | none: noise `(8, 5000)` | none: noise `(12, 5000)` | none: noise `(12, 5000)` |
| Training stages | 1 | 2 (VQ-VAE, then prior on frozen codes) | 1 (adversarial, 2 optimisers) | 1 | 1 |
| Loss | MSE + `kl_weight` · KL | MSE + VQ loss; then cross-entropy | WGAN-GP critic loss; −D(G(z)) | min-SNR-weighted MSE on v | MSE on velocity x₁ − x₀ |
| Sampling cost | 1 decoder pass | 625 sequential prior steps + 1 decoder pass | 1 generator pass | 50 UNet passes (DDIM; × 2 with guidance) | 10–50 UNet passes (Euler / Heun) |
| Output range | unbounded | unbounded | `tanh`, [−1, 1] | unbounded | unbounded |
| Conditioning | — | — | — | vector, tokens, dense; CFG | vector, tokens, dense; CFG |

## Not covered yet

Latent diffusion and latent flow matching (`src/ecgen/families/continuous/latent.py`) run the
same processes as the [Diffusion](diffusion.md) and [Flow Matching](flow_matching.md) pages in the
latent space of a frozen VAE or VQ-VAE; see [Diffusion and flow matching](../models/diffusion.md).
They do not have their own visual page yet.
