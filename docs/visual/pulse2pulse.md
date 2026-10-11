# Pulse2Pulse GAN — Visual Guide

Pulse2Pulse is a WaveGAN-style GAN. It turns noise **with the same shape as an ECG** into an ECG,
using a 1-D U-Net generator. A large convolutional critic scores real and generated ECGs, and
training uses the WGAN-GP objective. Code: `src/ecgen/families/gan/pulse2pulse.py`.
Written reference: [Pulse2Pulse GAN](../models/gan.md).

## Adversarial setup

<figure class="arch">
<svg viewBox="0 0 760 260" role="img" aria-label="Pulse2Pulse adversarial setup">
<g data-info="<b>z</b> — torch.empty(B, 8, 5000).uniform_(−1, 1), drawn fresh for every critic and generator step."><rect x="10" y="150" width="150" height="70" rx="6" class="tensor"/><path data-ecg="16,154,138,40" data-kind="noise" data-seed="3" class="trace noise"/><text x="85" y="212" class="xs c mono muted">z ~ U(−1,1) (B,8,5000)</text></g>
<path d="M162,185 L196,185" class="edge flow"/>
<g data-info="<b>WaveGANGenerator</b> (netG) — 1-D U-Net, about 10.5 M parameters. See the architecture below."><rect x="198" y="160" width="120" height="50" rx="6" class="module"/><text x="258" y="182" class="sm c">Generator G</text><text x="258" y="198" class="xs c muted">U-Net · tanh</text></g>
<path d="M320,185 L352,185" class="edge flow"/>
<path data-ecg="356,160,150,50" data-lead="I" data-seed="12" data-kind="fake" class="trace fake draw"/>
<text x="431" y="230" class="xs c muted">fake G(z)</text>
<path data-ecg="356,30,150,50" data-lead="I" data-seed="2" class="trace"/>
<text x="431" y="100" class="xs c muted">real x (first 8 leads)</text>
<path d="M510,55 L560,110" class="edge flow"/>
<path d="M510,185 L560,140" class="edge flow"/>
<g data-info="<b>WaveGANDiscriminator</b> (netD, the critic) — 7 strided convs with phase shuffle and a linear layer, about 204 M parameters. Outputs an unbounded score, no sigmoid."><rect x="562" y="95" width="100" height="60" rx="6" class="module"/><text x="612" y="120" class="sm c">Critic D</text><text x="612" y="137" class="xs c muted">score ∈ ℝ</text></g>
<path d="M664,112 L700,96" class="edge loss"/>
<path d="M664,138 L700,154" class="edge loss"/>
<rect x="702" y="78" width="54" height="30" rx="5" class="loss"/><text x="729" y="98" class="sm c">D(x)</text>
<rect x="702" y="142" width="54" height="30" rx="5" class="loss"/><text x="729" y="162" class="sm c">D(G(z))</text>
<text x="380" y="252" class="xs c muted">The critic learns to push D(x) up and D(G(z)) down; the generator learns to push D(G(z)) up.</text>
</svg>
<div class="arch-info">Hover over a block for details.</div>
</figure>

## Generator: a 1-D U-Net

Six strided convolutions (kernel 25) compress the noise from length 5000 down to 2. The decoder
upsamples it back, and at four resolutions it concatenates the encoder features of the same length
(skip connections). Hover over a layer:

<figure class="arch">
<svg data-diagram="unet" viewBox="0 0 760 420" role="img" aria-label="WaveGANGenerator U-Net"></svg>
<div class="arch-info">Defaults: <code>model_size = 50</code>, <code>num_channels = 8</code>. The total stride is 2·5·2·5·5·5 = 2500, so <code>seq_length</code> must be a multiple of 2500.</div>
</figure>

## Critic: `WaveGANDiscriminator`

<figure class="arch">
<svg data-diagram="critic" viewBox="0 0 780 250" role="img" aria-label="WaveGANDiscriminator"></svg>
<div class="arch-info">Hover over a layer. The final Linear layer is created on the first forward pass, so its size follows the input length.</div>
</figure>

### Phase shuffle

After each of the first six critic convolutions, `PhaseShuffle(2)` shifts every example in the batch
by a random k ∈ {−2, …, 2} samples and fills the gap with reflection padding. This stops the critic
from judging ECGs by the exact sample-level phase of upsampling artefacts:

<figure class="arch" data-widget="phase">
<svg viewBox="0 0 760 200" role="img" aria-label="Phase shuffle demonstration"></svg>
<figcaption>One feature channel, 36 time steps. Each example in a batch gets its own k.</figcaption>
</figure>

## How the losses are computed

Training uses manual optimisation with two Adam optimisers (lr = 1e-4, betas = (0.5, 0.9)).

$$
\mathcal{L}_D = \underbrace{D(G(z)) - D(x)}_{\text{negative Wasserstein estimate}}
+ \lambda\, \mathbb{E}\big[(\lVert \nabla_{\hat{x}} D(\hat{x}) \rVert_2 - 1)^2\big],
\quad \hat{x} = \alpha x + (1-\alpha)\,G(z),\; \alpha \sim U(0,1),\; \lambda = 10
$$

$$
\mathcal{L}_G = -\,D(G(z))
$$

<figure class="arch">
<svg viewBox="0 0 760 350" role="img" aria-label="WGAN-GP loss computation">
<text x="10" y="20" class="lg">Critic step</text><text x="110" y="20" class="xs muted">every batch · G(z) is detached</text>
<rect x="10" y="36" width="70" height="30" rx="4" class="tensor"/><text x="45" y="56" class="sm c">x</text>
<rect x="10" y="120" width="70" height="30" rx="4" class="tensor"/><text x="45" y="140" class="sm c">sg[G(z)]</text>
<g data-info="<b>Interpolate</b> — α ~ U(0, 1), one value per example, broadcast over leads and time: x̂ = α·x + (1 − α)·G(z). Try the slider below."><rect x="120" y="78" width="96" height="30" rx="4" class="ok"/><text x="168" y="98" class="sm c">x̂ = mix(α)</text></g>
<path d="M82,51 L118,86" class="edge"/>
<path d="M82,135 L118,100" class="edge"/>
<rect x="260" y="30" width="60" height="140" rx="6" class="module"/><text x="290" y="104" class="sm c">D</text>
<path d="M82,46 L258,46" class="edge flow"/>
<path d="M82,140 L258,140" class="edge flow"/>
<path d="M218,93 L258,93" class="edge flow"/>
<path d="M322,46 L370,46" class="edge loss"/>
<path d="M322,140 L370,140" class="edge loss"/>
<path d="M322,93 L370,93" class="edge loss"/>
<g data-info="d_real = netD(real).mean() over the batch."><rect x="372" y="32" width="110" height="28" rx="4" class="loss"/><text x="427" y="51" class="sm c">− mean D(x)</text></g>
<g data-info="d_fake = netD(fake).mean(); the fake batch is detached so the critic step does not update G."><rect x="372" y="126" width="110" height="28" rx="4" class="loss"/><text x="427" y="145" class="sm c">+ mean D(G(z))</text></g>
<g data-info="<b>calc_gradient_penalty</b> — autograd.grad of D(x̂) w.r.t. x̂ (create_graph=True), flattened per example, ((‖g‖₂ − 1)²).mean() × lmbda (10). Keeps the critic approximately 1-Lipschitz."><rect x="372" y="78" width="200" height="30" rx="4" class="loss"/><text x="472" y="98" class="sm c">+ λ (‖∇ₓ̂ D(x̂)‖₂ − 1)²</text></g>
<circle cx="610" cy="93" r="15" class="loss"/><text x="610" y="98" class="lg c">+</text>
<path d="M484,46 L598,82" class="edge loss"/>
<path d="M484,140 L598,104" class="edge loss"/>
<path d="M574,93 L593,93" class="edge loss"/>
<path d="M626,93 L656,93" class="edge loss"/>
<g data-info="<b>d_loss</b> → manual_backward → opt_d.step(). Also logged: train/d_wasserstein = D(x) − D(G(z))."><rect x="658" y="76" width="92" height="34" rx="5" class="loss"/><text x="704" y="97" class="sm c">ℒ_D → opt_d</text></g>
<path d="M656,104 C600,200 380,200 322,160" class="edge grad"/>
<text x="500" y="196" class="xs c" style="fill:var(--arch-grad)">updates D only</text>
<line x1="10" y1="222" x2="750" y2="222" class="gridline"/>
<text x="10" y="246" class="lg">Generator step</text><text x="130" y="246" class="xs muted">when (batch_idx + 1) % n_critic == 0</text>
<rect x="10" y="266" width="80" height="34" rx="4" class="tensor"/><text x="50" y="288" class="sm c">new z</text>
<path d="M92,283 L126,283" class="edge flow"/>
<rect x="128" y="264" width="80" height="38" rx="5" class="module"/><text x="168" y="288" class="sm c">G</text>
<path d="M210,283 L258,283" class="edge flow"/>
<rect x="260" y="264" width="60" height="38" rx="5" class="module"/><text x="290" y="288" class="sm c">D</text>
<path d="M322,283 L370,283" class="edge loss"/>
<g data-info="<b>g_loss</b> = −netD(fake).mean(). Gradients pass through D into G, but only opt_g steps, so D is not changed."><rect x="372" y="268" width="140" height="30" rx="4" class="loss"/><text x="442" y="288" class="sm c">− mean D(G(z))</text></g>
<path d="M514,283 L656,283" class="edge loss"/>
<rect x="658" y="266" width="92" height="34" rx="5" class="loss"/><text x="704" y="288" class="sm c">ℒ_G → opt_g</text>
<path d="M656,296 C560,330 230,330 196,304" class="edge grad"/>
<text x="430" y="342" class="xs c" style="fill:var(--arch-grad)">through D (not stepped) into G</text>
</svg>
<div class="arch-info">Hover over the loss terms.</div>
</figure>

### The gradient-penalty interpolation

The penalty is evaluated at random points on straight lines between real and generated ECGs:

<figure class="arch" data-widget="gp">
<svg viewBox="0 0 760 250" role="img" aria-label="Gradient penalty interpolation"></svg>
<div class="arch-controls"><label>α <input type="range" name="alpha" min="0" max="1" step="0.01" value="0.5"><output name="alpha"></output></label></div>
<figcaption>α = 1 gives the real ECG, α = 0 the generated one. During training every example in the batch gets its own α.</figcaption>
</figure>

### Update schedule

<figure class="arch" data-widget="schedule">
<svg viewBox="0 0 760 150" role="img" aria-label="n_critic schedule"></svg>
<div class="arch-controls"><label>n_critic <input type="range" name="ncritic" min="1" max="8" step="1" value="5"><output name="ncritic"></output></label></div>
<div class="arch-info"></div>
</figure>

Validation does not update anything. It logs `val/d_wasserstein = D(x) − D(G(z))` and
`val_loss = −val/d_wasserstein`, which is used for checkpointing.
