# GAN Models

Generative Adversarial Network models for ECG generation.

## Pulse2Pulse

Pulse2Pulse is a GAN-based model for ECG generation.

### Architecture

- Generator: Transforms noise + conditions to ECG signals
- Discriminator: Distinguishes real from generated ECGs

### Usage

```python
from models.gan import Pulse2Pulse

# Load model
model = Pulse2Pulse.load_from_checkpoint("checkpoint.ckpt")

# Generate samples
samples = model.generate(num_samples=16)
```

### Training

See [Training Guide](../training/guide.md) for training instructions.

## Future GAN Models

Additional GAN architectures will be added here.
