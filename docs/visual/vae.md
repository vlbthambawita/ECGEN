# VAE — Visual Guide

`VAE1D` compresses a 12-lead ECG into a short **sequence** of Gaussian latent vectors and
reconstructs it. After training, decoding random draws from N(0, I) generates new ECGs.
Code: `src/ecgen/families/vae/`, loss: `src/ecgen/families/vae/losses.py`. Written reference: [VAE](../models/vae.md).

## Forward pass during training

<figure class="arch">
<svg viewBox="0 0 760 230" role="img" aria-label="VAE forward pass">
<text x="70" y="22" class="sm c">input x</text>
<path data-ecg="10,40,120,70" data-lead="II" data-seed="2" data-beats="2" class="trace"/>
<text x="70" y="128" class="xs c mono muted">(B, 12, 5000)</text>
<path d="M134,75 L162,75" class="edge flow"/>
<g data-info="<b>Encoder1D</b> — conv_in, 4 stages of residual blocks, 3 stride-2 downsamplings (8×), middle blocks, and a conv head with 2 × 8 output channels."><path d="M164,30 L254,52 L254,98 L164,120 Z" class="module"/><text x="207" y="79" class="sm c">Encoder</text></g>
<path d="M256,60 L292,48" class="edge flow"/>
<path d="M256,90 L292,104" class="edge flow"/>
<g data-info="<b>mean μ</b> — (B, 8, 625)."><rect x="294" y="30" width="56" height="34" rx="4" class="tensor"/><text x="322" y="51" class="sm c">μ</text></g>
<g data-info="<b>logvar</b> — (B, 8, 625), the log-variance log σ²."><rect x="294" y="88" width="56" height="34" rx="4" class="tensor"/><text x="322" y="109" class="sm c">log σ²</text></g>
<text x="322" y="140" class="xs c mono muted">(B, 8, 625)</text>
<path d="M352,47 L398,70" class="edge flow"/>
<path d="M352,105 L398,82" class="edge flow"/>
<g data-info="<b>Reparameterisation</b> — z = μ + exp(0.5·logvar) ⊙ ε. The randomness lives in ε, so μ and logvar stay differentiable."><circle cx="420" cy="76" r="22" class="module pulse"/><text x="420" y="81" class="sm c">z</text></g>
<g data-info="<b>ε ~ N(0, I)</b> — torch.randn_like(std); a fresh draw every step."><rect x="384" y="160" width="72" height="30" rx="15" class="plain"/><text x="420" y="179" class="xs c">ε ~ N(0, I)</text></g>
<path d="M420,158 L420,100" class="edge"/>
<path d="M444,76 L478,76" class="edge flow"/>
<g data-info="<b>Decoder1D</b> — conv_in, middle blocks, 4 stages with 3 stride-2 transposed convolutions, GroupNorm → SiLU → Conv1d(64 → 12, k=7). No output activation."><path d="M480,52 L570,30 L570,120 L480,98 Z" class="module"/><text x="525" y="79" class="sm c">Decoder</text></g>
<path d="M572,75 L602,75" class="edge flow"/>
<text x="680" y="22" class="sm c">reconstruction x̂</text>
<path data-ecg="620,40,120,70" data-lead="II" data-seed="2" data-beats="2" class="trace mix draw"/>
<text x="680" y="128" class="xs c mono muted">(B, 12, 5000)</text>
<text x="380" y="220" class="xs c muted">Time is compressed 8× (5000 → 625) and the 12 leads are mixed into 8 latent channels.</text>
</svg>
<div class="arch-info">Hover over a block for details.</div>
</figure>

## Inside the encoder and decoder

The VAE uses the shared backbone with `latent_channels = 8`. Hover over a block to see the layer:

<figure class="arch" data-widget="backbone" data-mode="vae">
<svg data-diagram="backbone" viewBox="0 0 760 278" role="img" aria-label="VAE encoder and decoder"></svg>
<div class="arch-info">Hover over a block for the exact layer. Thin blocks are the stride-2 down/up-sampling convolutions.</div>
</figure>

See [Shared backbone](index.md#shared-backbone-encoder1d-decoder1d) for the residual block and a
latent-length calculator.

## The reparameterisation trick

Sampling is not differentiable, so `VAE1D.reparameterize` moves the randomness into an input `ε`
that needs no gradient:

<figure class="arch">
<svg viewBox="0 0 760 200" role="img" aria-label="Reparameterisation trick">
<rect x="20" y="30" width="80" height="34" rx="4" class="tensor"/><text x="60" y="51" class="sm c">μ</text>
<rect x="20" y="104" width="80" height="34" rx="4" class="tensor"/><text x="60" y="125" class="sm c">log σ²</text>
<path d="M102,121 L158,121" class="edge flow"/>
<rect x="160" y="104" width="120" height="34" rx="4" class="plain"/><text x="220" y="125" class="sm c mono">exp(0.5·)</text>
<path d="M282,121 L336,121" class="edge flow"/>
<rect x="338" y="104" width="50" height="34" rx="4" class="tensor"/><text x="363" y="125" class="sm c">σ</text>
<rect x="320" y="160" width="86" height="30" rx="15" class="plain"/><text x="363" y="179" class="xs c">ε ~ N(0, I)</text>
<circle cx="450" cy="121" r="16" class="plain"/><text x="450" y="126" class="sm c">⊙</text>
<path d="M390,121 L432,121" class="edge flow"/>
<path d="M406,175 L450,175 L450,139" class="edge"/>
<circle cx="560" cy="80" r="16" class="plain"/><text x="560" y="85" class="sm c">+</text>
<path d="M466,115 L545,88" class="edge flow"/>
<path d="M102,47 L280,47 L545,74" class="edge flow"/>
<path d="M576,80 L640,80" class="edge flow"/>
<rect x="642" y="63" width="90" height="34" rx="4" class="tensor"/><text x="687" y="84" class="sm c">z</text>
<path d="M640,90 C560,150 200,160 104,138" class="edge grad"/>
<path d="M640,70 C560,10 200,20 104,40" class="edge grad"/>
<text x="560" y="168" class="xs c" style="fill:var(--arch-grad)">∂L/∂μ, ∂L/∂logvar</text>
<text x="412" y="179" class="xs muted">no gradient</text>
</svg>
<figcaption>z = μ + exp(0.5·log σ²) ⊙ ε. Gradients (purple) reach μ and log σ², and through them the encoder.</figcaption>
</figure>

## How the loss is computed

```python
total, recon, kl = vae_loss(recon, x, mean, logvar, kl_weight=1e-4)
```

$$
\mathcal{L} \;=\; \underbrace{\frac{1}{BCL}\sum (\hat{x}-x)^2}_{\texttt{recon\_loss}}
\;+\; \texttt{kl\_weight}\cdot
\underbrace{\frac{-\tfrac12\sum\left(1+\log\sigma^2-\mu^2-\sigma^2\right)}{B\cdot C\cdot L}}_{\texttt{kl\_loss}}
$$

<figure class="arch">
<svg viewBox="0 0 760 270" role="img" aria-label="VAE loss computation">
<rect x="10" y="30" width="70" height="32" rx="4" class="tensor"/><text x="45" y="50" class="sm c">x</text>
<rect x="10" y="80" width="70" height="32" rx="4" class="tensor"/><text x="45" y="100" class="sm c">x̂</text>
<rect x="10" y="160" width="70" height="32" rx="4" class="tensor"/><text x="45" y="180" class="sm c">μ</text>
<rect x="10" y="210" width="70" height="32" rx="4" class="tensor"/><text x="45" y="230" class="sm c">log σ²</text>
<path d="M82,46 L188,64" class="edge loss"/>
<path d="M82,96 L188,80" class="edge loss"/>
<g data-info="<b>recon_loss</b> = F.mse_loss(recon, x): mean squared error over all B·12·5000 values."><rect x="190" y="50" width="170" height="44" rx="6" class="loss"/><text x="275" y="69" class="sm c">MSE(x̂, x)</text><text x="275" y="85" class="xs c muted">mean over B·C·L</text></g>
<path d="M82,176 L188,192" class="edge loss"/>
<path d="M82,226 L188,208" class="edge loss"/>
<g data-info="<b>KL divergence</b> of N(μ, σ²) from N(0, 1), summed over every latent element (B·8·625)."><rect x="190" y="178" width="170" height="44" rx="6" class="loss"/><text x="275" y="197" class="sm c">−½ Σ(1 + log σ² − μ² − σ²)</text><text x="275" y="213" class="xs c muted">sum over latent</text></g>
<path d="M362,200 L398,200" class="edge loss"/>
<g data-info="The KL sum is divided by the <b>input</b> size B·C·L (B·12·5000), not the latent size, so it is on the same per-element scale as the MSE."><rect x="400" y="182" width="96" height="36" rx="6" class="loss"/><text x="448" y="205" class="sm c">÷ (B·C·L)</text></g>
<path d="M498,200 L524,200" class="edge loss"/>
<g data-info="<b>kl_weight</b> (VAEConfig, default 1e-4) — trades reconstruction quality against how close the latent stays to N(0, I). Try the slider below."><rect x="526" y="182" width="96" height="36" rx="6" class="loss"/><text x="574" y="205" class="sm c">× kl_weight</text></g>
<text x="448" y="240" class="xs c mono muted">kl_loss</text>
<path d="M362,72 L650,72 L662,122" class="edge loss"/>
<text x="500" y="64" class="xs c mono muted">recon_loss</text>
<path d="M622,200 L662,156" class="edge loss"/>
<circle cx="672" cy="139" r="16" class="loss"/><text x="672" y="144" class="lg c">+</text>
<path d="M688,139 L706,139" class="edge loss"/>
<g data-info="<b>total_loss</b> — backpropagated by Lightning; logged as train/total_loss and val/total_loss (val_loss is used for checkpointing)."><rect x="708" y="122" width="44" height="34" rx="6" class="loss"/><text x="730" y="144" class="sm c">ℒ</text></g>
</svg>
<div class="arch-info">Hover over a loss term. Logged metrics: <code>train/recon_loss</code>, <code>train/kl_loss</code>, <code>train/total_loss</code> and the <code>val/…</code> equivalents.</div>
</figure>

### What `kl_weight` does

The KL term pulls each ECG's posterior q(z | x) towards the prior N(0, I). The sketch below is
**schematic** (one latent dimension, four ECGs); it is not computed from a trained model.

<figure class="arch" data-widget="kl">
<svg viewBox="0 0 760 230" role="img" aria-label="Effect of kl_weight on posteriors"></svg>
<div class="arch-controls"><label>kl_weight <input type="range" min="-6" max="0" step="0.5" value="-4"><output></output></label></div>
<figcaption>Small weight: posteriors are narrow and far apart, so reconstructions are sharp but random z from N(0, I) may land between them. Large weight: posteriors collapse onto the prior, so samples are smooth but blurry.</figcaption>
</figure>

## Sampling

`VAELightning.sample(n_samples, seq_length=5000)` skips the encoder:

<figure class="arch">
<svg viewBox="0 0 760 120" role="img" aria-label="VAE sampling">
<rect x="20" y="36" width="170" height="48" rx="6" class="tensor"/><text x="105" y="57" class="sm c">z ~ N(0, I)</text><text x="105" y="74" class="xs c mono muted">(n, 8, seq_length // 8)</text>
<path d="M192,60 L280,60" class="edge flow"/>
<circle r="4" cy="60" class="packet"><animateMotion dur="1.6s" repeatCount="indefinite" path="M192,0 L280,0"/></circle>
<path d="M282,36 L372,14 L372,106 L282,84 Z" class="module"/><text x="327" y="64" class="sm c">Decoder</text>
<path d="M374,60 L440,60" class="edge flow"/>
<path data-ecg="450,20,290,80" data-lead="V4" data-seed="6" data-beats="3" class="trace draw"/>
</svg>
<figcaption>No-grad, eval mode. The latent length is computed from the encoder's downsampling factor.</figcaption>
</figure>
