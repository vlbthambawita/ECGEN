# Flow Matching — Visual Guide

Flow matching (rectified flow) connects each noise sample to a data sample by a **straight line**
and trains the network to predict the direction of that line, the velocity. Sampling follows the
learned velocity field from noise (t = 0) to an ECG (t = 1) with a few Euler or Heun steps.
Code: `src/ecgen/processes/flow.py` (`RectifiedFlowProcess`), family
`src/ecgen/families/continuous/module.py` (`FlowMatchingModule`).
Written reference: [Diffusion and flow matching — Flow matching](../models/diffusion.md#flow-matching).

`FlowMatchingModule` is `DiffusionModule` with a different process. The backbone (`UNet1d` by
default), the conditioning, classifier-free guidance and EMA are identical; see the
[Diffusion visual guide](diffusion.md) for those.

## One training step

<figure class="arch">
<svg viewBox="0 0 760 230" role="img" aria-label="Flow matching training step">
<text x="65" y="22" class="sm c">noise x₀</text>
<g data-info="<b>x₀</b> = torch.randn_like(x₁) — note the naming: in flow.py x₀ is the noise and x₁ the data."><path data-ecg="10,30,110,50" data-kind="noise" data-seed="3" class="trace noise"/></g>
<text x="65" y="94" class="xs c mono muted">N(0, I)</text>
<text x="65" y="124" class="sm c">data x₁</text>
<g data-info="<b>x₁</b> — a batch of training ECGs, (B, 12, 5000)."><path data-ecg="10,132,110,50" data-lead="II" data-seed="2" data-beats="2" class="trace"/></g>
<text x="65" y="196" class="xs c mono muted">(B, 12, 5000)</text>
<path d="M124,58 L154,92" class="edge flow"/>
<path d="M124,156 L154,116" class="edge flow"/>
<g data-info="<b>interpolate</b> — x_t = t·x₁ + (1 − (1 − σ_min)·t)·x₀. With the default σ_min = 0 this is the straight line (1 − t)·x₀ + t·x₁. t ~ sigmoid(N(0, 1)) per example (logit-normal)."><circle cx="170" cy="104" r="16" class="plain"/><text x="170" y="109" class="lg c">+</text></g>
<text x="170" y="74" class="xs c muted">(1 − t) · x₀</text>
<text x="170" y="142" class="xs c muted">t · x₁</text>
<text x="212" y="152" class="xs mono muted">t ~ logit-N(0, 1)</text>
<path d="M188,104 L208,104" class="edge flow"/>
<text x="268" y="62" class="sm c">x_t</text>
<g data-info="<b>x_t</b> — a point on the straight path. At t = 0.5 it is half noise, half ECG."><path data-ecg="212,74,112,60" data-kind="noise" data-seed="8" class="trace mix"/></g>
<path d="M328,104 L352,104" class="edge flow"/>
<g data-info="<b>UNet1d(x_t, 1000·t, cond)</b> — the same backbone as diffusion. The network time input is t · time_scale (1000), the range its sinusoidal embedding was tuned for."><rect x="354" y="78" width="110" height="52" rx="6" class="module"/><text x="409" y="100" class="sm c">UNet1d</text><text x="409" y="117" class="xs c mono muted">x_t, 1000·t, c</text></g>
<g data-info="<b>CompositeConditioner</b> — same as diffusion; flow_matching_mimic.yaml conditions on 9 machine measurements (fourier encoder) with dropout 0.15."><rect x="354" y="168" width="110" height="34" rx="6" class="module"/><text x="409" y="189" class="sm c">conditioner</text></g>
<path d="M409,166 L409,132" class="edge flow"/>
<path d="M466,104 L490,104" class="edge flow"/>
<text x="550" y="62" class="sm c">v̂</text>
<g data-info="<b>v̂</b> — predicted velocity, (B, 12, 5000)."><path data-ecg="494,74,112,60" data-kind="noise" data-seed="12" class="trace fake"/></g>
<path d="M610,104 L634,104" class="edge loss"/>
<g data-info="<b>Loss</b> — plain MSE to x₁ − (1 − σ_min)·x₀; weight = 1 for every example."><rect x="636" y="80" width="114" height="48" rx="6" class="loss"/><text x="693" y="100" class="sm c">MSE</text><text x="693" y="116" class="xs c muted">target x₁ − x₀</text></g>
<text x="380" y="224" class="xs c muted">schematic traces · the module, backbone and loss code are shared with diffusion</text>
</svg>
<div class="arch-info">Hover over a block for details.</div>
</figure>

## Straight paths and solver steps

The left panel shows training pairs: each noise point is joined to a data point by a straight line,
and the network learns the slope x₁ − x₀ at every point of it. Because many lines pass through the
same region, the network learns their **average** direction, so the paths it follows at sampling
time (right) are curved. Euler steps follow them in straight segments; more steps or Heun stay
closer to the exact path. Minibatch OT pairing (`ot_coupling: true`) shortens the training paths
and makes them cross less.

<figure class="arch" data-widget="flowpaths">
<svg viewBox="0 0 760 350" role="img" aria-label="Straight-line paths and Euler steps"></svg>
<div class="arch-controls">
<label>t <input type="range" name="t" min="0" max="1" step="0.01" value="0"><output name="t"></output></label>
<button type="button" data-pairing="random">random pairing (default)</button><button type="button" data-pairing="ot">minibatch OT</button>
<label>steps <input type="range" name="steps" min="1" max="50" step="1" value="3"><output name="steps"></output></label>
<button type="button" data-solver="euler">Euler</button><button type="button" data-solver="heun">Heun</button>
</div>
<div class="arch-info"></div>
<figcaption>Toy 2-D example: noise N(0, I), data = three Gaussian clusters, and the exact velocity field of that toy instead of a trained network. A real sample is a point in 12 × 5000 = 60 000 dimensions. The OT pairing here is a 2-opt approximation; the code uses <code>scipy.optimize.linear_sum_assignment</code>.</figcaption>
</figure>

## Which t is trained on

`t_sampling: logit_normal` (default) draws t = sigmoid(n), n ~ N(`logit_mean` = 0,
`logit_std` = 1). It puts most training examples at intermediate t, where predicting the velocity is
hardest, and few near pure noise or clean data.

<figure class="arch">
<svg data-diagram="tsampling" viewBox="0 0 760 190" role="img" aria-label="Logit-normal time distribution"></svg>
</figure>

## How the loss is computed

```python
# RectifiedFlowProcess.training_pair(x1, noise)
if self.ot_coupling and x1.shape[0] > 1:
    noise = noise[ot_pairing(x1, noise)]      # re-pair noise with data inside the batch
t = self.sample_t(x1.shape[0], x1.device)     # logit-normal in (0, 1)
xt = t * x1 + (1 - (1 - self.sigma_min) * t) * noise
target = x1 - (1 - self.sigma_min) * noise
# then, as for diffusion: loss = (weight * ((backbone(xt, 1000 * t, cond) - target) ** 2).mean()).mean(), weight = 1
```

$$
x_t = t\,x_1 + \big(1 - (1-\sigma_{\min})\,t\big)\,x_0,\qquad
\frac{dx_t}{dt} = x_1 - (1-\sigma_{\min})\,x_0,\qquad
\mathcal{L} = \mathbb{E}_{t,\,x_0,\,x_1}\,\big\|\hat v(x_t, t, c) - (x_1 - (1-\sigma_{\min})\,x_0)\big\|^2
$$

<figure class="arch">
<svg viewBox="0 0 760 270" role="img" aria-label="Flow matching loss and gradient flow">
<path d="M682,168 L700,168 L700,34 L392,34 L392,58" class="edge grad"/>
<path d="M426,102 L426,122" class="edge grad"/>
<text x="540" y="26" class="xs c" style="fill:var(--arch-grad)">∂ℒ → UNet1d and conditioner</text>
<g data-info="<b>x₁</b> — clean batch (B, 12, 5000)."><rect x="20" y="62" width="90" height="34" rx="5" class="tensor"/><text x="65" y="83" class="sm c">data x₁</text></g>
<g data-info="<b>x₀</b> = torch.randn_like(x₁)."><rect x="20" y="164" width="90" height="34" rx="5" class="tensor"/><text x="65" y="185" class="sm c">noise x₀</text></g>
<path d="M112,79 L138,90" class="edge flow"/>
<path d="M112,181 L138,170" class="edge flow"/>
<g data-info="<b>RectifiedFlowProcess.training_pair(x₁, x₀)</b> — optional OT re-pairing, t ~ logit-normal, the point on the straight path and the velocity target. No parameters. The network sees t · 1000."><rect x="140" y="50" width="170" height="176" rx="6" class="plain"/><text x="225" y="72" class="sm c">training_pair</text>
<text x="225" y="100" class="xs c mono">[x₀ ← x₀[OT perm]]</text>
<text x="225" y="124" class="xs c mono">t = sigmoid(N(0, 1))</text>
<text x="225" y="148" class="xs c mono">x_t = t·x₁ + (1−t)·x₀</text>
<text x="225" y="172" class="xs c mono">target = x₁ − x₀</text>
<text x="225" y="196" class="xs c mono">weight = 1</text>
<text x="225" y="214" class="xs c muted">(σ_min = 0)</text></g>
<path d="M312,80 L338,80" class="edge flow"/>
<g data-info="<b>UNet1d</b> — trainable, (x_t, 1000·t, cond) → v̂."><rect x="340" y="60" width="104" height="40" rx="5" class="module"/><text x="392" y="77" class="sm c">UNet1d</text><text x="392" y="92" class="xs c mono muted">x_t, 1000·t, c</text></g>
<g data-info="<b>Conditioner</b> — trainable (conditional models only)."><rect x="340" y="124" width="104" height="32" rx="5" class="module"/><text x="392" y="144" class="sm c">conditioner</text></g>
<path d="M392,122 L392,102" class="edge flow"/>
<path d="M446,80 L476,80" class="edge flow"/>
<rect x="478" y="62" width="50" height="36" rx="5" class="tensor"/><text x="503" y="85" class="sm c">v̂</text>
<path d="M530,80 L622,80 L622,166" class="edge flow"/>
<path d="M312,200 L556,200" class="edge loss"/>
<text x="440" y="194" class="xs c muted">target x₁ − x₀</text>
<g data-info="<b>per_sample</b> = ((v̂ − target)²).flatten(1).mean(1); ℒ = mean over the batch (weight 1). loss_on / clean_observed work as for diffusion."><rect x="558" y="168" width="124" height="44" rx="6" class="loss"/><text x="620" y="187" class="sm c">MSE per sample</text><text x="620" y="202" class="xs c mono muted">(B,)</text></g>
<path d="M620,214 L620,230" class="edge loss"/>
<g data-info="<b>ℒ</b> — logged as train/loss and val_loss; the EMA of the weights is updated after each step, as for diffusion."><rect x="580" y="232" width="80" height="30" rx="6" class="loss"/><text x="620" y="252" class="sm c">ℒ (mean)</text></g>
<text x="300" y="258" class="xs c muted">purple = gradient path · the target is a constant</text>
</svg>
<div class="arch-info">Hover over a block.</div>
</figure>

## Sampling

`RectifiedFlowProcess.sample` starts from x = noise at t = 0 and takes `steps` equal steps of size
h = 1 / steps up to t = 1. Guidance is applied to the velocity with the same `cfg_model_fn`.

```python
for i in range(steps):                     # t = i / steps
    d = v(x, t)                            # u + w (c - u) with guidance
    if solver == "heun" and i < steps - 1:
        x_euler = x + h * d
        x = x + h * 0.5 * (d + v(x_euler, t + h))
    else:                                  # Euler (and the last Heun step)
        x = x + h * d
```

The cost of a sample is the number of backbone calls. Every call is the same `UNet1d` pass, so the
bars compare directly (defaults: `sample_steps = 50` for diffusion; `flow_matching_mimic.yaml` uses
25 Heun steps):

<figure class="arch">
<svg viewBox="0 0 760 230" role="img" aria-label="Network calls per sample">
<text x="196" y="20" class="xs r muted">sampler</text><text x="206" y="20" class="xs muted">network calls per sample (1 call = 1 UNet1d pass on (n, 12, 5000))</text>
<g data-info="Rectified flow, 10 Euler steps: 10 calls."><text x="196" y="46" class="sm r">flow · Euler 10</text><rect x="206" y="34" width="50" height="16" rx="2" class="ok"/><text x="262" y="46" class="xs mono">10</text></g>
<g data-info="Rectified flow, 25 Euler steps: 25 calls."><text x="196" y="72" class="sm r">flow · Euler 25</text><rect x="206" y="60" width="125" height="16" rx="2" class="ok"/><text x="337" y="72" class="xs mono">25</text></g>
<g data-info="Rectified flow, 25 Heun steps (flow_matching_mimic.yaml): 2 calls per step except the last, which is an Euler step: 2·24 + 1 = 49."><text x="196" y="98" class="sm r">flow · Heun 25</text><rect x="206" y="86" width="245" height="16" rx="2" class="ok"/><text x="457" y="98" class="xs mono">49</text></g>
<g data-info="DDIM, 50 steps (sample_steps default): 50 calls."><text x="196" y="124" class="sm r">diffusion · DDIM 50</text><rect x="206" y="112" width="250" height="16" rx="2" class="tensor"/><text x="462" y="124" class="xs mono">50</text></g>
<g data-info="Either model with classifier-free guidance (w ≠ 1): conditional + unconditional pass per step, 2 × 50 = 100."><text x="196" y="150" class="sm r">DDIM 50 + guidance</text><rect x="206" y="138" width="500" height="16" rx="2" class="tensor"/><text x="712" y="150" class="xs mono">100</text></g>
<g data-info="DDIM 50 with RePaint inpainting (jumps = 5): 5 calls for each of the first 49 steps + 1 for the last = 246."><text x="196" y="176" class="sm r">DDIM 50 + RePaint</text><rect x="206" y="164" width="500" height="16" rx="2" class="tensor" style="stroke-dasharray:4 3"/><text x="712" y="176" class="xs mono">246 →</text></g>
<text x="380" y="210" class="xs c muted">bar length ∝ calls (RePaint bar truncated) · fewer calls is faster; sample quality per step count is not shown here</text>
</svg>
<div class="arch-info">Hover over a bar for how the count is made up.</div>
</figure>

**Inpainting.** Flows support `inpaint="replace"`: before every step the observed entries are set to
the point on the straight path from the *initial* noise to the observation,
t·obs + (1 − t)·noise, and to `obs` itself at the end. RePaint is diffusion-only and raises an
error here. Models trained with `clean_observed` use the clean observation instead.
`x_init` / `t_start` start mid-path for SDEdit-style editing.
