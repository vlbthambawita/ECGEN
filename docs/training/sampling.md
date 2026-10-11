# Sampling with `ecgen sample`

Every checkpoint written by `ecgen train` records its family and the training
data contract (lead names, sampling rate, length, scaling, condition
normalizers), so one call samples any model:

```python
import ecgen

x = ecgen.sample("runs/diffusion_mimic_conditional/seed_42/checkpoints/last.ckpt", 16,
                 cond={"measurements": [800, 40, 128, 170, 258, 520, 60, 30, 40]},  # raw units
                 guidance=2.0, steps=50, seed=0,
                 output_dir="samples/", format="csv", plot=2)
# x: [16, 12, 5000] float32
```

```bash
ecgen sample runs/.../last.ckpt -n 16 -o samples/ --format csv \
    --cond measurements=800,40,128,170,258,520,60,30,40 --guidance 2 --steps 50 --seed 0 --plot 2
ecgen sample runs/.../last.ckpt -n 8 --cond report="Sinus tachycardia" --format wfdb
```

| Argument | Meaning |
|---|---|
| `cond` | Raw condition values: numeric fields in their original units (`NaN` = unknown), class indices, multi-hot vectors, strings for text fields. A value with a leading dimension of `n` (an `[n, F]` array, `n` class indices, `n` strings) gives one condition per sample; anything else is used for all samples. Fields left out use the model's null embedding. |
| `obs`, `obs_mask` | Partially observed ECGs and masks (1 = observed), e.g. only lead I, for lead reconstruction / imputation |
| `guidance` | Classifier-free guidance scale (conditional diffusion, flow matching, PixelCNN prior) |
| `steps` | Sampling steps (diffusion / flow) |
| `seed` | Reproducible sampling |
| `derive_12` | Complete the 8 independent leads (I, II, V1–V6) to the standard 12 (Einthoven / Goldberger) |
| `output_dir`, `format` | `npy` (one array), `csv` (one file per ECG, lead-name header), `wfdb` (`.hea`/`.dat`), or `None`; `samples.json` records leads, fs, units, conditions |
| `plot` | Render the first N samples as interactive HTML (`ecgen.plot_ecg`) |
| `family` | Registry name, only for checkpoints that predate family recording |

Other keyword arguments go to the model's `sample()` (e.g. `eta`, `solver`,
`inpaint="repaint"`, `temperature`). Units are `mV` for data trained with
`scale: none`; z-scored models output normalized units (`NU`).

`ecgen.load_model(checkpoint)` returns the model itself, and `hf://<repo>/<path>`
checkpoints are downloaded as for `ecgen.generate`. `ecgen.generate` (the
original Pulse2Pulse `.pt` loader) is unchanged.
