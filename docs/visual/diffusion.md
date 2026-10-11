# Diffusion (DDPM) — Visual Guide

A diffusion model learns to undo noise. In training, a clean ECG is mixed with Gaussian noise at a
random level `t` and the network learns to predict what was added (here: the *velocity* `v`). At
sampling time it starts from pure noise and removes it in about 50 DDIM steps.
Code: `src/ecgen/families/continuous/module.py` (`DiffusionModule`), process
`src/ecgen/processes/ddpm.py`, guidance `src/ecgen/processes/base.py`, backbone
`src/ecgen/nn/backbones/unet1d.py`, conditioning `src/ecgen/conditioning/composite.py`, EMA
`src/ecgen/training/ema.py`. Written reference: [Diffusion and flow matching](../models/diffusion.md).

## One training step

<figure class="arch">
<svg viewBox="0 0 760 230" role="img" aria-label="Diffusion training step">
<text x="65" y="22" class="sm c">clean x₀</text>
<g data-info="<b>x₀</b> — a batch of training ECGs, (B, 12, 5000), z-scored per lead (scale: zscore)."><path data-ecg="10,30,110,50" data-lead="II" data-seed="2" data-beats="2" class="trace"/></g>
<text x="65" y="94" class="xs c mono muted">(B, 12, 5000)</text>
<text x="65" y="124" class="sm c">noise ε</text>
<g data-info="<b>ε = torch.randn_like(x₀)</b> — fresh Gaussian noise with the same shape as the ECG."><path data-ecg="10,132,110,50" data-kind="noise" data-seed="3" class="trace noise"/></g>
<text x="65" y="196" class="xs c mono muted">N(0, I)</text>
<path d="M124,58 L154,92" class="edge flow"/>
<path d="M124,156 L154,116" class="edge flow"/>
<g data-info="<b>q_sample</b> — x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε with t drawn uniformly from {0, …, 999} per example. ᾱ_t comes from the cosine schedule (see the slider below)."><circle cx="170" cy="104" r="16" class="plain"/><text x="170" y="109" class="lg c">+</text></g>
<text x="170" y="74" class="xs c muted">√ᾱ_t · x₀</text>
<text x="170" y="142" class="xs c muted">√(1−ᾱ_t) · ε</text>
<text x="170" y="164" class="xs c mono muted">t ~ U{0…999}</text>
<path d="M188,104 L208,104" class="edge flow"/>
<text x="268" y="62" class="sm c">x_t</text>
<g data-info="<b>x_t</b> — the noisy ECG the network sees, (B, 12, 5000). Near t = 0 it is almost clean; near t = 999 it is almost pure noise."><path data-ecg="212,74,112,60" data-kind="noise" data-seed="8" class="trace mix"/></g>
<path d="M328,104 L352,104" class="edge flow"/>
<g data-info="<b>UNet1d(x_t, t, cond)</b> — the backbone (see below). t enters through a sinusoidal time embedding; the condition (if any) through AddToTimeEmb, cross-attention and input concatenation."><rect x="354" y="78" width="110" height="52" rx="6" class="module"/><text x="409" y="100" class="sm c">UNet1d</text><text x="409" y="117" class="xs c mono muted">(x_t, t, cond)</text></g>
<g data-info="<b>CompositeConditioner</b> — encodes the batch's condition fields (measurements, text, partial ECG, …) into vector / tokens / dense slots, with per-field dropout to learned null embeddings during training."><rect x="354" y="168" width="110" height="34" rx="6" class="module"/><text x="409" y="189" class="sm c">conditioner</text></g>
<path d="M409,166 L409,132" class="edge flow"/>
<text x="416" y="152" class="xs muted">cond</text>
<path d="M466,104 L490,104" class="edge flow"/>
<text x="550" y="62" class="sm c">v̂</text>
<g data-info="<b>v̂</b> — network output, (B, 12, 5000). With the default prediction = &quot;v&quot; it estimates v = √ᾱ·ε − √(1−ᾱ)·x₀."><path data-ecg="494,74,112,60" data-kind="noise" data-seed="12" class="trace fake"/></g>
<path d="M610,104 L634,104" class="edge loss"/>
<g data-info="<b>Loss</b> — per-example MSE between v̂ and the target v, times the min-SNR weight w(t), averaged over the batch. See “How the loss is computed”."><rect x="636" y="80" width="114" height="48" rx="6" class="loss"/><text x="693" y="100" class="sm c">w(t) · MSE</text><text x="693" y="116" class="xs c muted">target v</text></g>
<text x="380" y="224" class="xs c muted">schematic traces · shapes for the default 12-lead, 5000-sample data</text>
</svg>
<div class="arch-info">Hover over a block for details.</div>
</figure>

## The forward (noising) process

`DDPMProcess` precomputes `alphas_cumprod` ᾱ<sub>t</sub> for `num_steps = 1000`. Drag `t` to
see how much of the ECG survives. The plot shows ᾱ<sub>t</sub> for the default cosine
schedule and the linear one, plus the min-SNR loss weight for v-prediction (γ = 5).

<figure class="arch" data-widget="noising">
<svg viewBox="0 0 760 270" role="img" aria-label="Forward noising with the cosine schedule"></svg>
<div class="arch-controls">
<label>t <input type="range" name="t" min="0" max="999" step="1" value="0"><output name="t"></output></label>
<button type="button" data-schedule="cosine">cosine (default)</button><button type="button" data-schedule="linear">linear</button>
</div>
<div class="arch-info"></div>
<figcaption>The ᾱ<sub>t</sub> values are computed exactly as in <code>_schedule()</code>; the ECG is schematic. The linear schedule destroys the signal much earlier, so many of its 1000 steps are spent on near-pure noise.</figcaption>
</figure>

## Architecture: `UNet1d`

The default backbone is an ADM-style 1-D U-Net with `base_channels = 64`,
`channel_mults = [1, 2, 4, 4]`, `num_res_blocks = 2` and attention at level 3. The three
stride-2 convolutions take 5000 → 2500 → 1250 → 625, so the length must be divisible by 8.

<figure class="arch">
<svg data-diagram="unet1d" viewBox="0 0 760 500" role="img" aria-label="UNet1d block diagram"></svg>
<div class="arch-info">Hover over a block for the exact layers. Every skip tensor is concatenated on the channel axis before the next up-path ResBlock.</div>
</figure>

### Each ResBlock: where time and condition enter

<figure class="arch">
<svg viewBox="0 0 760 170" role="img" aria-label="UNet1d ResBlock with AdaGN">
<g data-info="Input h: (B, C_in, L)."><rect x="10" y="66" width="40" height="34" rx="4" class="tensor"/><text x="30" y="87" class="sm c">h</text></g>
<path d="M52,83 L66,83" class="edge flow"/>
<g data-info="GroupNorm(8, C_in) (norm1)."><rect x="68" y="66" width="52" height="34" rx="4" class="module"/><text x="94" y="87" class="sm c">GN</text></g>
<path d="M122,83 L132,83" class="edge"/>
<g data-info="SiLU."><rect x="134" y="66" width="46" height="34" rx="4" class="module"/><text x="157" y="87" class="sm c">SiLU</text></g>
<path d="M182,83 L192,83" class="edge"/>
<g data-info="conv1 = Conv1d(C_in → C_out, k=3, padding=1)."><rect x="194" y="66" width="66" height="34" rx="4" class="module"/><text x="227" y="87" class="sm c">Conv k=3</text></g>
<path d="M262,83 L272,83" class="edge"/>
<g data-info="<b>AdaGN</b> — GroupNorm(8, C_out), then FiLM: h · (1 + γ) + β with [γ, β] = Linear(256 → 2·C_out)(emb). The Linear is zero-initialised, so it starts as a plain GroupNorm. emb = time embedding (+ condition vector)."><rect x="274" y="62" width="78" height="42" rx="4" class="ok"/><text x="313" y="81" class="sm c">AdaGN</text><text x="313" y="95" class="xs c muted">γ, β from emb</text></g>
<path d="M354,83 L364,83" class="edge"/>
<g data-info="SiLU."><rect x="366" y="66" width="46" height="34" rx="4" class="module"/><text x="389" y="87" class="sm c">SiLU</text></g>
<path d="M414,83 L424,83" class="edge"/>
<g data-info="nn.Dropout(dropout); dropout = 0.0 by default."><rect x="426" y="66" width="60" height="34" rx="4" class="module"/><text x="456" y="87" class="sm c">Dropout</text></g>
<path d="M488,83 L498,83" class="edge"/>
<g data-info="conv2 = Conv1d(C_out → C_out, k=3, padding=1), <b>zero-initialised</b>: a fresh ResBlock is the identity (plus shortcut)."><rect x="500" y="62" width="84" height="42" rx="4" class="module"/><text x="542" y="81" class="sm c">Conv k=3</text><text x="542" y="95" class="xs c muted">zero init</text></g>
<path d="M586,83 L610,83" class="edge flow"/>
<circle cx="626" cy="83" r="14" class="ok"/><text x="626" y="88" class="lg c">+</text>
<path d="M642,83 L662,83" class="edge flow"/>
<g data-info="Output (B, C_out, L). At attention levels it then goes through SelfAttention1d and (with tokens) CrossAttention1d, both residual with zero-initialised output projections."><rect x="664" y="66" width="86" height="34" rx="4" class="tensor"/><text x="707" y="81" class="sm c">out</text><text x="707" y="94" class="xs c muted">→ attn (lvl 3)</text></g>
<path d="M30,64 L30,24 L626,24 L626,67" class="edge skip"/>
<g data-info="<b>Shortcut</b> — identity when C_in = C_out, otherwise Conv1d(C_in → C_out, k=1)."><rect x="230" y="12" width="210" height="24" rx="12" class="ok"/><text x="335" y="28" class="xs c">shortcut: identity or 1×1 conv</text></g>
<g data-info="<b>emb</b> (B, 256) = TimestepEmbedding(t) [+ AddToTimeEmb(c.vector)], shared by every ResBlock of the network."><rect x="254" y="130" width="118" height="30" rx="4" class="tensor"/><text x="313" y="149" class="sm c">emb (B, 256)</text></g>
<path d="M313,128 L313,106" class="edge flow"/>
</svg>
<div class="arch-info">Hover over a layer. The time step and the global condition vector act only through AdaGN.</div>
</figure>

### Conditioning: three slots

`CompositeConditioner` turns each condition field into one of three slots. During training each
field is replaced by a learned *null* embedding with its own probability (`conditioning.dropout`),
so the same network also learns the unconditional model needed for guidance.

<figure class="arch">
<svg viewBox="0 0 760 250" role="img" aria-label="Condition slots and null embeddings">
<text x="60" y="20" class="xs c muted">field</text><text x="210" y="20" class="xs c muted">encoder</text><text x="400" y="20" class="xs c muted">dropped? → null</text><text x="564" y="20" class="xs c muted">slot</text><text x="690" y="20" class="xs c muted">used by UNet1d</text>
<g data-info="E.g. machine measurements (B, 9), age, sex, labels."><rect x="10" y="34" width="100" height="40" rx="5" class="tensor"/><text x="60" y="51" class="sm c">measurements</text><text x="60" y="66" class="xs c muted">age, labels, …</text></g>
<path d="M112,54 L144,54" class="edge flow"/>
<g data-info="numeric / fourier / categorical / multilabel encoders (vector slot), then Linear(out_dim → vector_dim = 256). Several vector fields are <b>summed</b>."><rect x="146" y="34" width="128" height="40" rx="5" class="module"/><text x="210" y="51" class="sm c">encoder + Linear</text><text x="210" y="66" class="xs c mono muted">→ 256</text></g>
<path d="M276,54 L330,54" class="edge flow"/>
<g data-info="torch.where(dropped, null_vector[field], encoded). null_vector is a learned parameter, initialised to zeros. A field missing from the batch is always replaced by its null."><rect x="332" y="34" width="136" height="40" rx="5" class="ok"/><text x="400" y="51" class="sm c">null_vector</text><text x="400" y="66" class="xs c muted">learned, zero init</text></g>
<path d="M470,54 L500,54" class="edge flow"/>
<rect x="502" y="36" width="124" height="36" rx="5" class="tensor"/><text x="564" y="58" class="xs c mono">vector (B,256)</text>
<path d="M628,54 L642,54" class="edge flow"/>
<text x="646" y="51" class="xs">AddToTimeEmb</text><text x="646" y="64" class="xs muted">→ AdaGN</text>
<g data-info="E.g. a free-text report through the text / text_hash encoder with as_tokens = true, then Linear(→ token_dim = 256). Token fields are concatenated along N."><rect x="10" y="104" width="100" height="40" rx="5" class="tensor"/><text x="60" y="121" class="sm c">report text</text><text x="60" y="136" class="xs c muted">as_tokens</text></g>
<path d="M112,124 L144,124" class="edge flow"/>
<rect x="146" y="104" width="128" height="40" rx="5" class="module"/><text x="210" y="121" class="sm c">encoder + Linear</text><text x="210" y="136" class="xs c mono muted">→ (N, 256)</text>
<path d="M276,124 L330,124" class="edge flow"/>
<g data-info="A dropped row keeps its token count, but only slot 0 is valid and it holds null_token[field] (learned, init N(0, 0.02²)); the token mask hides the rest."><rect x="332" y="104" width="136" height="40" rx="5" class="ok"/><text x="400" y="121" class="sm c">null_token</text><text x="400" y="136" class="xs c muted">+ token mask</text></g>
<path d="M470,124 L500,124" class="edge flow"/>
<rect x="502" y="106" width="124" height="36" rx="5" class="tensor"/><text x="564" y="128" class="xs c mono">tokens (B,N,256)</text>
<path d="M628,124 L642,124" class="edge flow"/>
<text x="646" y="121" class="xs">cross-attention</text><text x="646" y="134" class="xs muted">level 3 + mid</text>
<g data-info="The field named obs reads the partially observed ECG and its mask from the batch (data.masks) or from sample(obs=, obs_mask=)."><rect x="10" y="174" width="100" height="40" rx="5" class="tensor"/><text x="60" y="191" class="sm c">obs, obs_mask</text><text x="60" y="206" class="xs c muted">partial ECG</text></g>
<path d="M112,194 L144,194" class="edge flow"/>
<g data-info="<b>masked_signal</b> — dense [x · mask, mask] = (B, 24, 5000) for 12 leads (optionally a 1×1 conv to <code>channels</code>)."><rect x="146" y="174" width="128" height="40" rx="5" class="module"/><text x="210" y="191" class="sm c">masked_signal</text><text x="210" y="206" class="xs c mono muted">[x·m, m]</text></g>
<path d="M276,194 L330,194" class="edge flow"/>
<g data-info="A dropped dense field is multiplied by 0: everything looks unobserved."><rect x="332" y="174" width="136" height="40" rx="5" class="ok"/><text x="400" y="191" class="sm c">zeros</text><text x="400" y="206" class="xs c muted">“nothing observed”</text></g>
<path d="M470,194 L500,194" class="edge flow"/>
<rect x="502" y="176" width="124" height="36" rx="5" class="tensor"/><text x="564" y="198" class="xs c mono">dense (B,24,5000)</text>
<path d="M628,194 L642,194" class="edge flow"/>
<text x="646" y="191" class="xs">concat to x_t</text><text x="646" y="204" class="xs muted">before conv_in</text>
<text x="380" y="240" class="xs c muted">training: field dropped with probability dropout[field] · sampling with guidance: drop="all" for the unconditional pass</text>
</svg>
<div class="arch-info">Hover over a block. The <code>ssm</code> backbone accepts vector and dense slots only; <code>unet1d</code> and <code>dit1d</code> accept all three.</div>
</figure>

### Other backbones

The process and the conditioning do not care which network predicts `v`. Two other backbones
take the same `forward(x, t, cond)`:

- **`dit1d`** (`src/ecgen/nn/backbones/dit1d.py`): cuts the signal into patches of 25 samples
  (5000 → 200 tokens of width 256), runs 8 transformer blocks with AdaLN-Zero modulation from the
  time + condition embedding and cross-attention to condition tokens in every block, then
  un-patches with a zero-initialised Linear.
- **`ssm`** (`src/ecgen/nn/backbones/ssm.py`): SSSD-style stack of 8 residual blocks with
  bidirectional S4D layers at full resolution (128 channels, no downsampling; the receptive
  field is the whole signal). `mamba` swaps S4D for bidirectional Mamba.

## Classifier-free guidance

With a conditional model and `guidance = w ≠ 1`, every sampling step runs the backbone twice: once
with the condition (`c`) and once with all fields replaced by their nulls (`u`,
`encode_condition(..., drop="all")`). `cfg_model_fn` combines them as **u + w·(c − u)**.

<figure class="arch" data-widget="cfg">
<svg viewBox="0 0 760 250" role="img" aria-label="Classifier-free guidance"></svg>
<div class="arch-controls">
<label>guidance w <input type="range" name="w" min="0" max="5" step="0.1" value="1"><output name="w"></output></label>
</div>
<div class="arch-info"></div>
<figcaption>Schematic traces shown as predicted clean ECGs. Guidance is applied to the raw network output v, but x̂₀ = √ᾱ·x<sub>t</sub> − √(1−ᾱ)·v is affine in v, so the picture is the same. Default <code>guidance = 1.0</code> (off).</figcaption>
</figure>

## How the loss is computed

```python
# ContinuousGenerativeModule._loss (simplified: no masks)
x0 = self.to_model_space(signal)                       # identity for DiffusionModule
cond = self.adapt_condition(self.condition(batch), x0.shape[-1])  # with per-field dropout
pair = self.process.training_pair(x0, torch.randn_like(x0))      # x_t, t, target, weight
out = self.backbone(pair.xt, pair.t, cond)
per_sample = ((out - pair.target) ** 2).flatten(1).mean(1)
loss = (pair.weight * per_sample).mean()
```

With the default `prediction = "v"` and `min_snr_gamma = 5`:

$$
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon,\qquad
v = \sqrt{\bar\alpha_t}\,\varepsilon - \sqrt{1-\bar\alpha_t}\,x_0,\qquad
t \sim \mathcal{U}\{0,\dots,999\}
$$

$$
\mathcal{L} = \frac{1}{B}\sum_{b} w(t_b)\,\operatorname{mean}\big[(\hat v_b - v_b)^2\big],\qquad
w(t) = \frac{\min(\mathrm{SNR}_t,\ \gamma)}{\mathrm{SNR}_t + 1},\quad
\mathrm{SNR}_t = \frac{\bar\alpha_t}{1-\bar\alpha_t}
$$

<figure class="arch">
<svg viewBox="0 0 760 290" role="img" aria-label="Diffusion loss and gradient flow">
<path d="M682,168 L700,168 L700,34 L392,34 L392,58" class="edge grad"/>
<path d="M426,102 L426,122" class="edge grad"/>
<text x="540" y="26" class="xs c" style="fill:var(--arch-grad)">∂ℒ → UNet1d and conditioner (the process has no parameters)</text>
<g data-info="<b>x₀</b> — clean batch (B, 12, 5000)."><rect x="20" y="62" width="90" height="34" rx="5" class="tensor"/><text x="65" y="83" class="sm c">x₀</text></g>
<g data-info="<b>ε</b> = torch.randn_like(x₀)."><rect x="20" y="164" width="90" height="34" rx="5" class="tensor"/><text x="65" y="185" class="sm c">ε ~ N(0, I)</text></g>
<path d="M112,79 L138,90" class="edge flow"/>
<path d="M112,181 L138,170" class="edge flow"/>
<g data-info="<b>DDPMProcess.training_pair(x₀, ε)</b> — draws t, builds x_t, the v-target and the min-SNR weight. Pure tensor arithmetic: no parameters, nothing to train. x_t and the target do not require grad."><rect x="140" y="50" width="170" height="196" rx="6" class="plain"/><text x="225" y="72" class="sm c">training_pair</text>
<text x="225" y="100" class="xs c mono">t ~ U{0, …, 999}</text>
<text x="225" y="124" class="xs c mono">x_t = √ᾱ x₀ + √(1−ᾱ) ε</text>
<text x="225" y="148" class="xs c mono">v = √ᾱ ε − √(1−ᾱ) x₀</text>
<text x="225" y="172" class="xs c mono">SNR = ᾱ / (1 − ᾱ)</text>
<text x="225" y="196" class="xs c mono">w = min(SNR, 5)/(SNR+1)</text></g>
<path d="M312,80 L338,80" class="edge flow"/>
<g data-info="<b>UNet1d</b> — trainable backbone, (x_t, t, cond) → v̂."><rect x="340" y="60" width="104" height="40" rx="5" class="module"/><text x="392" y="77" class="sm c">UNet1d</text><text x="392" y="92" class="xs c mono muted">(x_t, t, cond)</text></g>
<g data-info="<b>Conditioner</b> — trainable encoders, projections and null embeddings (only for conditional models)."><rect x="340" y="124" width="104" height="32" rx="5" class="module"/><text x="392" y="144" class="sm c">conditioner</text></g>
<path d="M392,122 L392,102" class="edge flow"/>
<path d="M446,80 L476,80" class="edge flow"/>
<rect x="478" y="62" width="50" height="36" rx="5" class="tensor"/><text x="503" y="85" class="sm c">v̂</text>
<path d="M530,80 L622,80 L622,166" class="edge flow"/>
<path d="M312,200 L556,200" class="edge loss"/>
<text x="440" y="194" class="xs c muted">target v</text>
<g data-info="<b>per_sample</b> = ((v̂ − v)²).flatten(1).mean(1) — mean over 12 × 5000 values. With loss_on = &quot;unobserved&quot; the mean is taken over unobserved entries only."><rect x="558" y="168" width="124" height="44" rx="6" class="loss"/><text x="620" y="187" class="sm c">MSE per sample</text><text x="620" y="202" class="xs c mono muted">(B,)</text></g>
<path d="M620,214 L620,236" class="edge loss"/>
<path d="M312,252 L604,252" class="edge loss"/>
<text x="440" y="246" class="xs c muted">weight w(t)  (B,)</text>
<circle cx="620" cy="252" r="14" class="loss"/><text x="620" y="257" class="sm c">×</text>
<path d="M634,252 L664,252" class="edge loss"/>
<g data-info="<b>ℒ</b> = (weight · per_sample).mean(), logged as train/loss and val_loss. After each optimiser step, on_train_batch_end updates the EMA copy of the backbone + conditioner (ema_decay 0.999)."><rect x="666" y="234" width="80" height="36" rx="6" class="loss"/><text x="706" y="257" class="sm c">ℒ (mean)</text></g>
<g data-info="<b>EMA</b> — shadow ← decay · shadow + (1 − decay) · param after every training batch (ema_decay = 0.999, 0.9995 in diffusion_mimic.yaml). Not trained by the loss; used for sampling and stored in checkpoints."><rect x="478" y="118" width="64" height="30" rx="5" class="frozen"/><text x="510" y="137" class="sm c">EMA</text></g>
<path d="M446,96 L476,124" class="edge noarrow" style="stroke-dasharray:3 3"/>
<text x="380" y="284" class="xs c muted">purple = gradient path · orange = loss terms · the target and weight are constants (no gradient)</text>
</svg>
<div class="arch-info">Hover over a block.</div>
</figure>

!!! note "Partial-ECG options"
    With `data.masks`, `loss_on: unobserved` averages the error only where `obs_mask = 0`, and
    `clean_observed: true` (SSSD-style) overwrites the observed entries of x<sub>t</sub> with the
    clean signal before the backbone, in training and at every sampling step.

## Sampling

`sample()` swaps in the EMA weights, encodes the condition (and its null version for guidance),
then lets `DDPMProcess.sample` run DDIM from pure noise over `steps` evenly spaced time steps
(default 50 of the 1000).

<figure class="arch">
<svg viewBox="0 0 760 230" role="img" aria-label="Diffusion sampling pipeline">
<g data-info="<b>ema.swap()</b> — copy the averaged weights into the backbone and conditioner for the duration of sampling (use_ema=True), then restore the training weights."><rect x="10" y="20" width="120" height="40" rx="6" class="frozen"/><text x="70" y="37" class="sm c">EMA weights</text><text x="70" y="51" class="xs c muted">ema.swap()</text></g>
<path d="M132,40 L160,40" class="edge flow"/>
<g data-info="<b>encode_condition</b> — c from the given condition values; if guidance ≠ 1, also u with drop=&quot;all&quot; (every field → its null embedding)."><rect x="162" y="20" width="190" height="40" rx="6" class="module"/><text x="257" y="37" class="sm c">encode c, and u (drop="all")</text><text x="257" y="51" class="xs c muted">only for conditional models</text></g>
<path d="M354,40 L382,40" class="edge flow"/>
<g data-info="<b>x_T</b> = randn(n, 12, 5000) (from a seeded generator if seed= is given)."><rect x="384" y="20" width="150" height="40" rx="6" class="tensor"/><text x="459" y="37" class="sm c">x ~ N(0, I)</text><text x="459" y="51" class="xs c mono muted">(n, 12, 5000)</text></g>
<path d="M459,62 L459,86" class="edge flow"/>
<g data-info="<b>DDPMProcess.sample</b> — for each t in timesteps(steps) (evenly spaced from 999 down to 0): one DDIM update. The final step returns the predicted x̂₀ directly."><rect x="40" y="88" width="560" height="104" rx="8" class="plain"/>
<text x="56" y="108" class="sm">for t, t′ in zip(timesteps, timesteps[1:] + [−1]):</text>
<text x="76" y="128" class="xs">out = u + w·(c − u)            # cfg_model_fn; just c when w = 1</text>
<text x="76" y="145" class="xs">x̂₀ = √ᾱ_t·x − √(1−ᾱ_t)·out ;  ε̂ = √(1−ᾱ_t)·x + √ᾱ_t·out</text>
<text x="76" y="162" class="xs">x = √ᾱ_t′·x̂₀ + √(1−ᾱ_t′−σ²)·ε̂ + σ·z,  σ = η·√((1−ᾱ_t′)/(1−ᾱ_t)·(1−ᾱ_t/ᾱ_t′))</text>
<text x="76" y="179" class="xs">if inpaint: x = m·q_sample(obs, t′) + (1 − m)·x</text></g>
<path d="M602,140 L632,140" class="edge flow"/>
<g data-info="<b>paste_observed</b> (default True) — copy the observed part of obs into the output: x = m·obs + (1 − m)·x."><rect x="634" y="118" width="116" height="44" rx="6" class="module"/><text x="692" y="137" class="sm c">paste observed</text><text x="692" y="152" class="xs c muted">if obs given</text></g>
<path d="M692,164 L692,180" class="edge flow"/>
<path data-ecg="620,182,130,40" data-lead="II" data-seed="5" data-beats="2" class="trace draw"/>
<text x="320" y="214" class="xs c muted">η = 0: deterministic DDIM (default) · η = 1: ancestral (DDPM-like) · x_init / start_step: SDEdit-style editing</text>
</svg>
<div class="arch-info">Hover over a block.</div>
</figure>

The widget runs that loop on one lead. Change the number of steps, η, and the inpainting mode:
**replace** overwrites the observed part (here the first 35 %) with the observation noised to the
next level after every step; **RePaint** (`inpaint="repaint"`, `jumps=5`) additionally diffuses
back from t′ to t and redoes the step, 5 times per step, so the generated part can adjust to the
observed part.

<figure class="arch" data-widget="ddim">
<svg viewBox="0 0 760 270" role="img" aria-label="DDIM sampling loop"></svg>
<div class="arch-controls">
<label>steps <input type="range" name="steps" min="5" max="100" step="5" value="50"><output name="steps"></output></label>
<label>η <input type="range" name="eta" min="0" max="1" step="0.1" value="0"><output name="eta"></output></label>
<button type="button" data-inpaint="none">no inpainting</button><button type="button" data-inpaint="replace">replace</button><button type="button" data-inpaint="repaint">RePaint</button>
<button type="button" data-action="play">Pause</button><button type="button" data-action="restart">Restart</button>
</div>
<div class="arch-info"></div>
<figcaption>Toy: the “denoiser” is a hand-made function that knows the target ECG, so this shows the update rule, not model quality. Early steps predict a blurry average; detail appears as ᾱ<sub>t</sub> → 1.</figcaption>
</figure>

| Option | Where | Effect |
|---|---|---|
| `steps` | `sample(steps=)`, default `sample_steps = 50` | network calls (× 2 with guidance, × `jumps` with RePaint) |
| `eta` | `sample(eta=)`, default 0 | 0 = deterministic DDIM, 1 = ancestral |
| `guidance` | `sample(guidance=)`, default 1.0 | classifier-free guidance scale w |
| `inpaint` | `"replace"`, `"repaint"`; `"clean"` is automatic for `clean_observed` models | training-free partial-ECG completion |
| `use_ema` | default True | sample with the EMA weights |

Flow matching uses the same module, backbone and conditioning with a different process; see
[Flow matching](flow_matching.md).
