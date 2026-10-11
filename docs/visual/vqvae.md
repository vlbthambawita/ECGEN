# VQ-VAE + PixelCNN Prior — Visual Guide

The VQ-VAE turns each ECG into a sequence of **discrete code indices** (stage 1), and a separate
autoregressive model learns which index sequences are likely (stage 2). Generation samples indices
from the prior and decodes them. Code: `src/ecgen/families/vqvae/`, losses: `src/ecgen/families/vqvae/losses.py`.
Written reference: [VQ-VAE + Prior](../models/vqvae.md).

## Two stages at a glance

<figure class="arch">
<svg viewBox="0 0 760 250" role="img" aria-label="VQ-VAE two-stage overview">
<text x="10" y="24" class="lg">Stage 1</text><text x="80" y="24" class="xs muted">VQVAELightning — learn the codebook, encoder and decoder</text>
<path data-ecg="10,40,90,40" data-lead="II" data-seed="2" data-beats="2" class="trace"/>
<path d="M104,60 L128,60" class="edge flow"/>
<rect x="130" y="42" width="80" height="36" rx="5" class="module"/><text x="170" y="64" class="sm c">Encoder</text>
<path d="M212,60 L236,60" class="edge flow"/>
<rect x="238" y="42" width="80" height="36" rx="5" class="module"/><text x="278" y="64" class="sm c">VQ</text>
<path d="M320,60 L344,60" class="edge flow"/>
<rect x="346" y="42" width="80" height="36" rx="5" class="module"/><text x="386" y="64" class="sm c">Decoder</text>
<path d="M428,60 L452,60" class="edge flow"/>
<path data-ecg="456,40,90,40" data-lead="II" data-seed="2" data-beats="2" class="trace mix"/>
<rect x="570" y="42" width="180" height="36" rx="5" class="loss"/><text x="660" y="64" class="sm c">MSE + VQ loss</text>
<text x="10" y="114" class="lg">Stage 2</text><text x="80" y="114" class="xs muted">PriorLightning — learn p(codes) with the stage-1 model frozen</text>
<path data-ecg="10,130,90,40" data-lead="II" data-seed="4" data-beats="2" class="trace"/>
<path d="M104,150 L128,150" class="edge flow"/>
<rect x="130" y="132" width="188" height="36" rx="5" class="frozen"/><text x="224" y="154" class="sm c">Encoder + VQ (frozen)</text>
<path d="M320,150 L344,150" class="edge flow"/>
<rect x="346" y="132" width="80" height="36" rx="5" class="tensor"/><text x="386" y="154" class="sm c mono">codes</text>
<path d="M428,150 L452,150" class="edge flow"/>
<rect x="454" y="132" width="100" height="36" rx="5" class="module"/><text x="504" y="154" class="sm c">PixelCNN</text>
<path d="M556,150 L568,150" class="edge flow"/>
<rect x="570" y="132" width="180" height="36" rx="5" class="loss"/><text x="660" y="154" class="sm c">cross-entropy</text>
<text x="10" y="204" class="lg">Sample</text><text x="80" y="204" class="xs muted">PriorLightning.sample()</text>
<rect x="130" y="212" width="80" height="34" rx="5" class="module"/><text x="170" y="233" class="sm c">PixelCNN</text>
<path d="M212,229 L236,229" class="edge flow"/>
<rect x="238" y="212" width="80" height="34" rx="5" class="tensor"/><text x="278" y="233" class="sm c mono">codes</text>
<path d="M320,229 L344,229" class="edge flow"/>
<rect x="346" y="212" width="80" height="34" rx="5" class="frozen"/><text x="386" y="233" class="sm c">Decoder</text>
<path d="M428,229 L452,229" class="edge flow"/>
<path data-ecg="456,208,140,40" data-lead="II" data-seed="7" data-beats="3" class="trace draw"/>
</svg>
</figure>

## Stage 1: forward pass

<figure class="arch">
<svg viewBox="0 0 760 250" role="img" aria-label="VQ-VAE stage 1 forward pass">
<text x="60" y="22" class="sm c">input x</text>
<path data-ecg="6,40,108,64" data-lead="II" data-seed="2" data-beats="2" class="trace"/>
<text x="60" y="122" class="xs c mono muted">(B, 12, 5000)</text>
<path d="M116,72 L136,72" class="edge flow"/>
<g data-info="<b>Encoder1D</b> with latent_channels = 64. It outputs 128 channels split into (mean, logvar); VQVAE1D.encode keeps only mean as z."><path d="M138,28 L218,50 L218,94 L138,116 Z" class="module"/><text x="176" y="76" class="sm c">Encoder</text></g>
<path d="M220,62 L250,62" class="edge flow"/>
<path d="M220,84 L250,98" class="edge ghost"/>
<text x="254" y="108" class="xs muted">logvar ✕</text>
<g data-info="<b>z</b> — continuous latent (B, 64, 625): one 64-d vector per time step."><rect x="252" y="40" width="44" height="44" rx="4" class="tensor"/><text x="274" y="67" class="sm c">z</text></g>
<path d="M298,62 L326,62" class="edge flow"/>
<g data-info="<b>VectorQuantizer(512, 64, commitment_cost=0.25)</b> — L2 distance from every z_t to all 512 codebook vectors, argmin, look up the winner."><rect x="328" y="30" width="96" height="64" rx="6" class="module"/><text x="376" y="56" class="sm c">VQ</text><text x="376" y="74" class="xs c muted">codebook</text><text x="376" y="87" class="xs c mono muted">512 × 64</text></g>
<path d="M426,62 L452,62" class="edge flow"/>
<g data-info="<b>z_q</b> — quantized latent (B, 64, 625). In the forward pass it equals the codebook vectors; in the backward pass gradients are copied to z (straight-through)."><rect x="454" y="40" width="44" height="44" rx="4" class="tensor"/><text x="476" y="67" class="sm c">z_q</text></g>
<path d="M500,62 L524,62" class="edge flow"/>
<g data-info="<b>Decoder1D</b> — 64 × 625 → 12 × 5000."><path d="M526,50 L606,28 L606,116 L526,94 Z" class="module"/><text x="566" y="76" class="sm c">Decoder</text></g>
<path d="M608,72 L626,72" class="edge flow"/>
<text x="692" y="22" class="sm c">x̂</text>
<path data-ecg="630,40,124,64" data-lead="II" data-seed="2" data-beats="2" class="trace mix draw"/>
<text x="692" y="122" class="xs c mono muted">(B, 12, 5000)</text>
<path d="M376,96 L376,150" class="edge flow"/>
<g data-info="<b>indices</b> (B, 625) — the discrete representation of the ECG. Stage 2 learns their distribution."><rect x="250" y="152" width="252" height="34" rx="4" class="plain"/>
<rect x="254" y="156" width="26" height="26" rx="3" style="fill:hsl(30,65%,55%)"/><text x="267" y="174" class="xs c mono">37</text>
<rect x="284" y="156" width="26" height="26" rx="3" style="fill:hsl(90,65%,55%)"/><text x="297" y="174" class="xs c mono">5</text>
<rect x="314" y="156" width="26" height="26" rx="3" style="fill:hsl(90,65%,55%)"/><text x="327" y="174" class="xs c mono">5</text>
<rect x="344" y="156" width="26" height="26" rx="3" style="fill:hsl(210,65%,55%)"/><text x="357" y="174" class="xs c mono">212</text>
<rect x="374" y="156" width="26" height="26" rx="3" style="fill:hsl(300,65%,55%)"/><text x="387" y="174" class="xs c mono">88</text>
<rect x="404" y="156" width="26" height="26" rx="3" style="fill:hsl(150,65%,55%)"/><text x="417" y="174" class="xs c mono">401</text>
<rect x="434" y="156" width="26" height="26" rx="3" style="fill:hsl(30,65%,55%)"/><text x="447" y="174" class="xs c mono">37</text>
<text x="478" y="174" class="sm c mono muted">…</text></g>
<text x="376" y="204" class="xs c mono muted">indices (B, 625), values in [0, 512)</text>
<text x="380" y="236" class="xs c muted">forward() returns (recon, vq_loss, indices)</text>
</svg>
<div class="arch-info">Hover over a block for details.</div>
</figure>

The encoder and decoder are the shared backbone with `latent_channels = 64`:

<figure class="arch" data-widget="backbone" data-mode="vqvae">
<svg data-diagram="backbone" viewBox="0 0 760 278" role="img" aria-label="VQ-VAE encoder and decoder"></svg>
<div class="arch-info">Hover over a block for the exact layer. The faded logvar half is computed but not used.</div>
</figure>

## Vector quantization

Each latent vector z<sub>t</sub> is replaced by the nearest of the 512 codebook vectors
e<sub>k</sub>. The animation uses a 2-D toy space with 12 codes and 8 time steps:

<figure class="arch" data-widget="vq">
<svg viewBox="0 0 760 300" role="img" aria-label="Vector quantization animation"></svg>
<div class="arch-controls"><button type="button" data-action="resample">New latents</button></div>
<figcaption>1. encoder outputs z<sub>t</sub> (circles) · 2. find the nearest codebook vector (dashed) · 3. snap to it and record its index.</figcaption>
</figure>

## Stage 1: how the loss is computed

```python
recon, vq_loss, indices = vqvae(x)
total, recon_loss, vq_loss = vqvae_loss(recon, x, vq_loss)   # total = recon_loss + vq_loss
```

$$
\mathcal{L} = \underbrace{\operatorname{MSE}(\hat{x}, x)}_{\texttt{recon\_loss}}
+ \underbrace{\operatorname{MSE}(\operatorname{sg}[z_q],\, z)}_{\text{trains the encoder}}
+ \beta \cdot \underbrace{\operatorname{MSE}(z_q,\, \operatorname{sg}[z])}_{\text{trains the codebook}},
\qquad \beta = \texttt{commitment\_cost} = 0.25
$$

<figure class="arch">
<svg viewBox="0 0 760 330" role="img" aria-label="VQ-VAE loss and gradient flow">
<path d="M672,58 L672,30 L62,30 L62,58" class="edge grad"/>
<text x="560" y="22" class="xs c" style="fill:var(--arch-grad)">∂ recon_loss: decoder → z_q</text>
<g data-info="<b>Straight-through estimator</b> — quantized = z + (quantized − z).detach(). The forward value is z_q, but ∂z_q/∂z = I, so the decoder's gradient is copied unchanged to z, skipping the non-differentiable argmin, and flows on into the encoder."><rect x="200" y="12" width="176" height="20" rx="10" class="plain"/><text x="288" y="26" class="xs c" style="fill:var(--arch-grad)">straight-through copy → z</text></g>
<rect x="20" y="60" width="84" height="40" rx="5" class="module"/><text x="62" y="85" class="sm c">Encoder</text>
<path d="M106,80 L146,80" class="edge flow"/>
<rect x="148" y="62" width="40" height="36" rx="4" class="tensor"/><text x="168" y="85" class="sm c">z</text>
<path d="M190,80 L240,80" class="edge flow"/>
<g data-info="<b>argmin</b> over L2 distances is not differentiable, so no gradient flows through it."><rect x="242" y="60" width="90" height="40" rx="5" class="plain"/><text x="287" y="78" class="xs c">argmin ‖z − e‖</text><text x="287" y="92" class="xs c muted">+ lookup</text></g>
<g data-info="<b>Codebook</b> — nn.Embedding(512, 64), initialised uniform in [−1/512, 1/512]. Only the β term trains it."><rect x="242" y="134" width="90" height="36" rx="5" class="module"/><text x="287" y="156" class="sm c">codebook</text></g>
<path d="M287,132 L287,102" class="edge"/>
<path d="M334,80 L384,80" class="edge flow"/>
<rect x="386" y="62" width="44" height="36" rx="4" class="tensor"/><text x="408" y="85" class="sm c">z_q</text>
<path d="M432,80 L472,80" class="edge flow"/>
<rect x="474" y="60" width="84" height="40" rx="5" class="module"/><text x="516" y="85" class="sm c">Decoder</text>
<path d="M560,80 L600,80" class="edge flow"/>
<g data-info="<b>recon_loss</b> = F.mse_loss(recon, x)."><rect x="602" y="60" width="140" height="40" rx="5" class="loss"/><text x="672" y="85" class="sm c">MSE(x̂, x)</text></g>
<path d="M168,100 L150,226" class="edge loss"/>
<path d="M408,100 L420,226" class="edge loss"/>
<path d="M310,172 L360,226" class="edge loss"/>
<g data-info="<b>codebook_loss</b> in the code = F.mse_loss(quantized.detach(), z). The codebook side is detached, so this term only moves z (the encoder) towards its chosen code."><rect x="40" y="228" width="220" height="44" rx="6" class="loss"/><text x="150" y="247" class="sm c">MSE(sg[z_q], z)</text><text x="150" y="263" class="xs c muted">→ encoder only</text></g>
<path d="M44,226 C30,180 34,140 46,102" class="edge grad"/>
<g data-info="<b>commitment_loss</b> in the code = F.mse_loss(quantized, z.detach()), multiplied by commitment_cost = 0.25. z is detached, so this term only moves the selected codebook vectors towards the encoder outputs."><rect x="300" y="228" width="220" height="44" rx="6" class="loss"/><text x="410" y="247" class="sm c">β · MSE(z_q, sg[z])</text><text x="410" y="263" class="xs c muted">→ codebook only</text></g>
<path d="M306,226 C280,210 262,196 262,172" class="edge grad"/>
<path d="M150,274 L150,296 L600,296 L600,266" class="edge loss"/>
<path d="M522,250 L582,250" class="edge loss"/>
<path d="M672,102 L608,234" class="edge loss"/>
<circle cx="600" cy="250" r="16" class="loss"/><text x="600" y="255" class="lg c">+</text>
<path d="M616,250 L660,250" class="edge loss"/>
<rect x="662" y="232" width="80" height="36" rx="6" class="loss"/><text x="702" y="255" class="sm c">ℒ</text>
<text x="380" y="320" class="xs c muted">purple = gradient path · sg[·] = .detach() · the argmin itself passes no gradient</text>
</svg>
<div class="arch-info">Hover over the loss terms and the straight-through label.</div>
</figure>

!!! note "β is on the codebook term here"
    The original VQ-VAE paper puts β on the encoder (commitment) term. In
    `VectorQuantizer.forward`, β multiplies the term that updates the codebook, even though that
    variable is named `commitment_loss`. The diagram shows what the code does.

## Stage 2: the PixelCNN prior

`PriorLightning` loads the stage-1 checkpoint, freezes it, and turns every training ECG into its
625 code indices. `PixelCNNPrior` is then trained to predict each index from the ones before it.

<figure class="arch">
<svg viewBox="0 0 760 300" role="img" aria-label="PixelCNN prior training">
<rect x="10" y="20" width="110" height="40" rx="5" class="frozen"/><text x="65" y="38" class="sm c">VQ-VAE</text><text x="65" y="52" class="xs c muted">encode_to_indices</text>
<path d="M122,40 L150,40" class="edge flow"/>
<g data-info="<b>codes c</b> — (B, 625) integers. They are both the network input and the target."><rect x="152" y="22" width="90" height="36" rx="5" class="tensor"/><text x="197" y="45" class="sm c mono">c (B, 625)</text></g>
<path d="M244,40 L272,40" class="edge flow"/>
<g data-info="<b>nn.Embedding(512, 128)</b> → (B, 128, 625)."><rect x="274" y="22" width="96" height="36" rx="5" class="module"/><text x="322" y="45" class="sm c">Embedding</text></g>
<path d="M372,40 L400,40" class="edge flow"/>
<g data-info="<b>Shift right by one</b> — F.pad(x, (1, 0))[:, :, :-1]. Position t now holds the embedding of code t−1 (zeros at t = 0), so nothing can see code t itself."><rect x="402" y="22" width="96" height="36" rx="5" class="plain"/><text x="450" y="45" class="sm c">shift → 1</text></g>
<path d="M500,40 L528,40 L528,86" class="edge flow"/>
<g data-info="<b>10 × CausalGatedBlock(128, k=2)</b> with dilations 1, 2, 4, …, 512. Each returns (x + residual(h), skip(h)). Receptive field: 1 + (1 + 2 + … + 512) = 1024 codes."><rect x="410" y="88" width="236" height="96" rx="6" class="module"/>
<text x="528" y="108" class="sm c">10 × CausalGatedBlock</text>
<rect x="424" y="120" width="18" height="50" rx="2" class="plain"/><text x="433" y="182" class="xs c mono muted">1</text>
<rect x="446" y="120" width="18" height="50" rx="2" class="plain"/><text x="455" y="182" class="xs c mono muted">2</text>
<rect x="468" y="120" width="18" height="50" rx="2" class="plain"/><text x="477" y="182" class="xs c mono muted">4</text>
<rect x="490" y="120" width="18" height="50" rx="2" class="plain"/><text x="499" y="182" class="xs c mono muted">8</text>
<rect x="512" y="120" width="18" height="50" rx="2" class="plain"/>
<rect x="534" y="120" width="18" height="50" rx="2" class="plain"/>
<rect x="556" y="120" width="18" height="50" rx="2" class="plain"/>
<rect x="578" y="120" width="18" height="50" rx="2" class="plain"/>
<rect x="600" y="120" width="18" height="50" rx="2" class="plain"/>
<rect x="622" y="120" width="18" height="50" rx="2" class="plain"/><text x="631" y="182" class="xs c mono muted">512</text></g>
<path d="M528,186 L528,214" class="edge flow"/>
<text x="540" y="204" class="xs muted">Σ skips</text>
<g data-info="<b>Head</b> — ReLU → Conv1d(128, 128, 1) → ReLU → Conv1d(128, 512, 1) on the summed skip outputs."><rect x="452" y="216" width="152" height="34" rx="5" class="module"/><text x="528" y="238" class="sm c">head (1×1 convs)</text></g>
<path d="M450,233 L360,233" class="edge flow"/>
<g data-info="<b>logits</b> — (B, 512, 625). Column t is a distribution over the 512 codes for position t."><rect x="258" y="216" width="100" height="34" rx="5" class="tensor"/><text x="308" y="238" class="sm c mono">logits</text></g>
<path d="M256,233 L190,233" class="edge loss"/>
<g data-info="<b>prior_loss</b> = F.cross_entropy over all B·625 positions, with the <b>unshifted</b> codes as targets. Logged as train/loss and val/loss."><rect x="40" y="214" width="148" height="40" rx="6" class="loss"/><text x="114" y="232" class="sm c">cross-entropy</text><text x="114" y="247" class="xs c muted">target = c (not shifted)</text></g>
<path d="M197,60 C197,120 120,150 114,212" class="edge loss"/>
<text x="150" y="140" class="xs muted">targets</text>
</svg>
<div class="arch-info">Hover over a block. Use <code>PriorLightning.compute_loss(indices)</code>; never shift the targets yourself — the shift is inside the model.</div>
</figure>

### Each gated block

<figure class="arch">
<svg viewBox="0 0 760 170" role="img" aria-label="CausalGatedBlock">
<rect x="10" y="66" width="50" height="34" rx="4" class="tensor"/><text x="35" y="88" class="sm c">x</text>
<path d="M62,83 L86,83" class="edge flow"/>
<g data-info="Left-pad by (k − 1)·d = d zeros so position t only sees t and t − d."><rect x="88" y="66" width="90" height="34" rx="4" class="plain"/><text x="133" y="82" class="xs c">left pad d</text><text x="133" y="94" class="xs c muted">(causal)</text></g>
<path d="M180,83 L204,83" class="edge flow"/>
<g data-info="Conv1d(128 → 256, k=2, dilation=d), split into a filter half and a gate half."><rect x="206" y="66" width="110" height="34" rx="4" class="module"/><text x="261" y="88" class="sm c">conv k=2, d</text></g>
<path d="M318,76 L350,56" class="edge flow"/>
<path d="M318,90 L350,110" class="edge flow"/>
<rect x="352" y="40" width="64" height="30" rx="4" class="module"/><text x="384" y="60" class="sm c">tanh</text>
<rect x="352" y="96" width="64" height="30" rx="4" class="module"/><text x="384" y="116" class="sm c">σ</text>
<path d="M418,55 L446,76" class="edge flow"/>
<path d="M418,111 L446,90" class="edge flow"/>
<circle cx="462" cy="83" r="15" class="plain"/><text x="462" y="88" class="sm c">⊙</text>
<path d="M477,83 L504,83" class="edge flow"/>
<text x="490" y="74" class="xs c">h</text>
<path d="M506,83 L530,50 L554,50" class="edge flow"/>
<path d="M506,83 L530,120 L554,120" class="edge flow"/>
<rect x="556" y="34" width="84" height="32" rx="4" class="module"/><text x="598" y="54" class="sm c">1×1 res</text>
<rect x="556" y="104" width="84" height="32" rx="4" class="module"/><text x="598" y="124" class="sm c">1×1 skip</text>
<path d="M642,50 L676,50" class="edge flow"/>
<circle cx="690" cy="50" r="13" class="ok"/><text x="690" y="55" class="sm c">+</text>
<path d="M35,64 L35,14 L690,14 L690,35" class="edge skip"/>
<path d="M704,50 L752,50" class="edge flow"/><text x="730" y="42" class="xs c">x′</text>
<path d="M642,120 L700,120" class="edge flow"/><text x="726" y="124" class="xs c">skip</text>
</svg>
<div class="arch-info">tanh(W<sub>f</sub> ∗ x) ⊙ σ(W<sub>g</sub> ∗ x): the sigmoid gate decides how much of the filter output passes. Hover over the padding and the conv.</div>
</figure>

### Why logit t never sees code t

Hover over (or tap) a top node to see which codes it depends on. This toy stack has three layers with
dilations 1, 2, 4; the real prior has ten (1 … 512).

<figure class="arch" data-widget="causal">
<svg viewBox="0 0 780 300" role="img" aria-label="Causal receptive field"></svg>
<div class="arch-info"></div>
</figure>

## Sampling

`PixelCNNPrior.sample()` fills the 625 positions left to right. At step t it runs the network on the
codes so far, takes the logits at t, divides by `temperature`, and draws one code. Then the frozen
VQ-VAE decodes all indices at once. Lower temperatures make the most likely code win more often.

<figure class="arch" data-widget="ar">
<svg viewBox="0 0 760 300" role="img" aria-label="Autoregressive sampling"></svg>
<div class="arch-controls">
<label>temperature <input type="range" name="temperature" min="0.2" max="2" step="0.1" value="1"><output name="temperature"></output></label>
<button type="button" data-action="play">Pause</button><button type="button" data-action="restart">Restart</button>
</div>
<figcaption>The toy “prior” here prefers codes close to the previous one, so you can see the effect of temperature. Sampling 625 codes takes 625 network passes, which makes this the slowest generator of the three.</figcaption>
</figure>
