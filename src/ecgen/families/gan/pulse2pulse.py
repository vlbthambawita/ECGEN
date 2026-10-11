from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ecgen.conditioning import build_conditioner
from ecgen.families.base import Capabilities, GenerativeModule
from ecgen.nn.injection import FiLM
from ecgen.registry import FAMILIES
from ecgen.utils.config import resolve_config


class Transpose1dLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int = 11,
        upsample: Optional[int] = None,
        output_padding: int = 1,
    ) -> None:
        super().__init__()
        self.upsample = upsample
        self.upsample_layer = (
            nn.Upsample(scale_factor=upsample) if upsample is not None else None
        )
        reflection_pad = kernel_size // 2
        self.reflection_pad = nn.ConstantPad1d(reflection_pad, value=0)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.conv1d_transpose = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.upsample:
            assert self.upsample_layer is not None
            x = self.upsample_layer(x)
            x = self.reflection_pad(x)
            return self.conv1d(x)
        return self.conv1d_transpose(x)


class Transpose1dLayerMultiInput(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int = 11,
        upsample: Optional[int] = None,
        output_padding: int = 1,
    ) -> None:
        super().__init__()
        self.upsample = upsample
        self.upsample_layer = (
            nn.Upsample(scale_factor=upsample) if upsample is not None else None
        )
        reflection_pad = kernel_size // 2
        self.reflection_pad = nn.ConstantPad1d(reflection_pad, value=0)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.conv1d_transpose = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        if self.upsample:
            assert self.upsample_layer is not None
            x = torch.cat((x, skip), dim=1)
            x = self.upsample_layer(x)
            x = self.reflection_pad(x)
            return self.conv1d(x)
        return self.conv1d_transpose(x)


class WaveGANGenerator(nn.Module):
    """
    Pulse2Pulse generator from the original WaveGAN implementation, configured for 8‑lead ECG.
    """

    def __init__(
        self,
        model_size: int = 50,
        num_channels: int = 8,
        post_proc_filt_len: int = 0,
        verbose: bool = False,
        upsample: bool = True,
        extra_in_channels: int = 0,
        cond_dim: int = 0,
        output_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.model_size = model_size
        self.num_channels = num_channels
        self.output_scale = output_scale
        self.post_proc_filt_len = post_proc_filt_len
        self.verbose = verbose

        stride = 1 if upsample else 4
        upsample_val: Optional[int] = 5 if upsample else None

        self.deconv_1 = Transpose1dLayer(
            5 * model_size, 5 * model_size, 25, stride, upsample=upsample_val
        )
        self.deconv_2 = Transpose1dLayerMultiInput(
            5 * model_size * 2, 3 * model_size, 25, stride, upsample=upsample_val
        )
        self.deconv_3 = Transpose1dLayerMultiInput(
            3 * model_size * 2, model_size, 25, stride, upsample=upsample_val
        )
        self.deconv_5 = Transpose1dLayerMultiInput(
            model_size * 2, model_size // 2, 25, stride, upsample=2
        )
        self.deconv_6 = Transpose1dLayerMultiInput(
            model_size, model_size // 5, 25, stride, upsample=upsample_val
        )
        self.deconv_7 = Transpose1dLayer(
            model_size // 5, num_channels, 25, stride, upsample=2
        )

        # extra_in_channels: dense conditions concatenated to the input noise
        self.conv_1 = nn.Conv1d(
            num_channels + extra_in_channels,
            model_size // 5,
            25,
            stride=2,
            padding=25 // 2,
        )
        self.conv_2 = nn.Conv1d(
            model_size // 5, model_size // 2, 25, stride=5, padding=25 // 2
        )
        self.conv_3 = nn.Conv1d(
            model_size // 2, model_size, 25, stride=2, padding=25 // 2
        )
        self.conv_4 = nn.Conv1d(
            model_size, model_size * 3, 25, stride=5, padding=25 // 2
        )
        self.conv_5 = nn.Conv1d(
            model_size * 3, model_size * 5, 25, stride=5, padding=25 // 2
        )
        self.conv_6 = nn.Conv1d(
            model_size * 5, model_size * 5, 25, stride=5, padding=25 // 2
        )

        if post_proc_filt_len and post_proc_filt_len > 0:
            self.post_proc_filter = nn.Conv1d(
                num_channels, num_channels, post_proc_filt_len
            )
        else:
            self.post_proc_filter = None

        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)

        # cond_dim > 0: FiLM on every decoder stage (created after the init loop
        # so they stay zero-initialised)
        self.films = nn.ModuleList()
        if cond_dim > 0:
            for ch in (
                5 * model_size,
                3 * model_size,
                model_size,
                model_size // 2,
                model_size // 5,
            ):
                self.films.append(FiLM(cond_dim, ch))

    def _film(self, i: int, x: Tensor, c: Optional[Tensor]) -> Tensor:
        return self.films[i](x, c) if len(self.films) else x

    def forward(
        self, x: Tensor, c: Optional[Tensor] = None, dense: Optional[Tensor] = None
    ) -> Tensor:
        if dense is not None:
            x = torch.cat([x, dense], dim=1)
        conv_1_out = F.leaky_relu(self.conv_1(x))
        conv_2_out = F.leaky_relu(self.conv_2(conv_1_out))
        conv_3_out = F.leaky_relu(self.conv_3(conv_2_out))
        conv_4_out = F.leaky_relu(self.conv_4(conv_3_out))
        conv_5_out = F.leaky_relu(self.conv_5(conv_4_out))
        x = F.leaky_relu(self.conv_6(conv_5_out))

        x = F.relu(self._film(0, self.deconv_1(x), c))
        x = F.relu(self._film(1, self.deconv_2(x, conv_5_out), c))
        x = F.relu(self._film(2, self.deconv_3(x, conv_4_out), c))
        x = F.relu(self._film(3, self.deconv_5(x, conv_3_out), c))
        x = F.relu(self._film(4, self.deconv_6(x, conv_2_out), c))
        x = torch.tanh(self.deconv_7(x))
        if self.output_scale != 1.0:
            x = x * self.output_scale

        if self.post_proc_filter is not None:
            x = self.post_proc_filter(x)

        return x


class PhaseShuffle(nn.Module):
    """
    Phase shuffling as in WaveGAN: random circular shifts with reflection padding.
    """

    def __init__(self, shift_factor: int) -> None:
        super().__init__()
        self.shift_factor = shift_factor

    def forward(self, x: Tensor) -> Tensor:
        if self.shift_factor == 0:
            return x

        k_list = (
            torch.randint(
                low=0,
                high=2 * self.shift_factor + 1,
                size=(x.shape[0],),
                device=x.device,
            )
            - self.shift_factor
        )

        k_map: dict[int, list[int]] = {}
        for idx, k in enumerate(k_list.tolist()):
            k_map.setdefault(int(k), []).append(idx)

        x_shuffle = x.clone()
        for k, idxs in k_map.items():
            if k > 0:
                x_shuffle[idxs] = F.pad(x[idxs][..., :-k], (k, 0), mode="reflect")
            else:
                x_shuffle[idxs] = F.pad(x[idxs][..., -k:], (0, -k), mode="reflect")

        return x_shuffle


class WaveGANDiscriminator(nn.Module):
    """
    Pulse2Pulse discriminator with dynamic final linear layer to accommodate sequence length.
    """

    def __init__(
        self,
        model_size: int = 64,
        num_channels: int = 8,
        shift_factor: int = 2,
        alpha: float = 0.2,
        verbose: bool = False,
        extra_in_channels: int = 0,
        cond_dim: int = 0,
    ) -> None:
        super().__init__()
        self.model_size = model_size
        self.num_channels = num_channels
        self.alpha = alpha
        self.verbose = verbose

        # extra_in_channels: dense conditions seen next to the signal (paired D)
        self.conv1 = nn.Conv1d(
            num_channels + extra_in_channels, model_size, 25, stride=2, padding=11
        )
        self.conv2 = nn.Conv1d(model_size, 2 * model_size, 25, stride=2, padding=11)
        self.conv3 = nn.Conv1d(2 * model_size, 5 * model_size, 25, stride=2, padding=11)
        self.conv4 = nn.Conv1d(
            5 * model_size, 10 * model_size, 25, stride=2, padding=11
        )
        self.conv5 = nn.Conv1d(
            10 * model_size, 20 * model_size, 25, stride=4, padding=11
        )
        self.conv6 = nn.Conv1d(
            20 * model_size, 25 * model_size, 25, stride=4, padding=11
        )
        self.conv7 = nn.Conv1d(
            25 * model_size, 100 * model_size, 25, stride=4, padding=11
        )

        self.ps1 = PhaseShuffle(shift_factor)
        self.ps2 = PhaseShuffle(shift_factor)
        self.ps3 = PhaseShuffle(shift_factor)
        self.ps4 = PhaseShuffle(shift_factor)
        self.ps5 = PhaseShuffle(shift_factor)
        self.ps6 = PhaseShuffle(shift_factor)

        self.fc1: Optional[nn.Linear] = None

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)) and m is not self.fc1:
                nn.init.kaiming_normal_(m.weight.data)

        # Projection discriminator (Miyato & Koyama, 2018): adds
        # <embed(c), mean_t(features)> to the critic output
        self.projection = (
            nn.Linear(cond_dim, 100 * model_size, bias=False) if cond_dim > 0 else None
        )

    def _ensure_fc(self, x: Tensor) -> None:
        flat_dim = x.size(1) * x.size(2)
        if self.fc1 is None or getattr(self.fc1, "in_features", None) != flat_dim:
            self.fc1 = nn.Linear(flat_dim, 1, device=x.device)
            nn.init.kaiming_normal_(self.fc1.weight.data)

    def features(self, x: Tensor) -> Tensor:
        """Convolutional features ``[B, 100 * model_size, L']`` (input incl. dense channels)."""
        x = F.leaky_relu(self.conv1(x), negative_slope=self.alpha)
        x = self.ps1(x)
        x = F.leaky_relu(self.conv2(x), negative_slope=self.alpha)
        x = self.ps2(x)
        x = F.leaky_relu(self.conv3(x), negative_slope=self.alpha)
        x = self.ps3(x)
        x = F.leaky_relu(self.conv4(x), negative_slope=self.alpha)
        x = self.ps4(x)
        x = F.leaky_relu(self.conv5(x), negative_slope=self.alpha)
        x = self.ps5(x)
        x = F.leaky_relu(self.conv6(x), negative_slope=self.alpha)
        x = self.ps6(x)
        x = F.leaky_relu(self.conv7(x), negative_slope=self.alpha)
        return x

    def forward(
        self, x: Tensor, c: Optional[Tensor] = None, dense: Optional[Tensor] = None
    ) -> Tensor:
        if dense is not None:
            x = torch.cat([x, dense], dim=1)
        features = self.features(x)
        x = features

        x = x.view(x.size(0), -1, 1)
        self._ensure_fc(x)
        assert self.fc1 is not None
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        if self.projection is not None and c is not None:
            out = out + (self.projection(c) * features.mean(dim=-1)).sum(
                dim=1, keepdim=True
            )
        return out


def calc_gradient_penalty(
    netD: nn.Module,
    real_data: Tensor,
    fake_data: Tensor,
    lmbda: float,
) -> Tensor:
    """
    WGAN‑GP gradient penalty.
    """
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, device=real_data.device)
    alpha = alpha.expand_as(real_data)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)
    disc_interpolates = netD(interpolates)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates, device=real_data.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lmbda
    return gradient_penalty


@dataclass
class Pulse2PulseConfig:
    model_size: int = 50
    num_channels: int = 8
    seq_length: int = 5000
    lr: float = 1e-4
    b1: float = 0.5
    b2: float = 0.9
    lmbda: float = 10.0
    n_critic: int = 5
    # Conditional GAN (all off = the original Pulse2Pulse)
    conditioning: Optional[dict[str, Any]] = None  # see ecgen.conditioning
    l1_weight: float = (
        0.0  # pix2pix-style L1 to the real ECG (paired / dense conditions)
    )
    l1_on: str = "unobserved"  # "unobserved" (needs obs_mask) or "all"
    output_scale: float = (
        1.0  # generator output = output_scale * tanh(...); e.g. 8 for z-scored data
    )


@FAMILIES.register("pulse2pulse")
class Pulse2PulseGAN(GenerativeModule):
    """
    PyTorch Lightning wrapper around the Pulse2Pulse WaveGAN generator + discriminator.

    With ``config.conditioning`` it is a conditional WGAN-GP:

    - global conditions (``vector``) modulate the generator's decoder with FiLM and
      enter the critic through a projection head; the critic has its own copy of
      the conditioner (trained by the critic's optimizer, without dropout);
    - time-aligned conditions (``dense``, e.g. ``masked_signal``) are concatenated
      to the generator's input noise and to the critic's input, which makes it a
      paired (pix2pix-style) translator, e.g. 1 lead -> 8 leads; set ``l1_weight``
      (e.g. 100) to add an L1 term on the unobserved part.
    """

    family = "gan"
    supports = Capabilities(conditional=True)
    accepts = frozenset({"vector", "dense"})

    def __init__(
        self, config: Pulse2PulseConfig | dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        super().__init__()
        config = resolve_config(Pulse2PulseConfig, config, kwargs)
        self.save_hyperparameters(config.__dict__)
        self.config = config
        if config.l1_on not in ("all", "unobserved"):
            raise ValueError(
                f"l1_on must be 'all' or 'unobserved', got {config.l1_on!r}"
            )

        self.init_conditioning(config.conditioning)
        cond_dim = dense_channels = 0
        self.d_conditioner = None
        if self.conditioner is not None:
            cond_dim = (
                self.conditioner.vector_dim if "vector" in self.conditioner.slots else 0
            )
            dense_channels = self.conditioner.dense_channels
            d_spec = {k: v for k, v in config.conditioning.items() if k != "dropout"}
            self.d_conditioner = build_conditioner(d_spec)

        self.netG = WaveGANGenerator(
            model_size=config.model_size,
            num_channels=config.num_channels,
            extra_in_channels=dense_channels,
            cond_dim=cond_dim,
            output_scale=config.output_scale,
        )
        self.netD = WaveGANDiscriminator(
            model_size=config.model_size,
            num_channels=config.num_channels,
            extra_in_channels=dense_channels,
            cond_dim=cond_dim,
        )
        # Build netD.fc1 now: created lazily it would be missing from the
        # critic's optimizer (configure_optimizers runs before the first forward).
        with torch.no_grad():
            dummy = torch.zeros(
                1, config.num_channels + dense_channels, config.seq_length
            )
            self.netD._ensure_fc(self.netD.features(dummy).reshape(1, -1, 1))

        self.automatic_optimization = False

        self._val_real_sample: Tensor | None = None
        self._val_fake_sample: Tensor | None = None

    def forward(self, noise: Tensor) -> Tensor:
        return self.netG(noise)

    @property
    def fixed_seq_length(self) -> Optional[int]:
        return self.config.seq_length

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)
        # Older checkpoints may hold an fc1 of another size (lazy creation);
        # rebuild it at the saved size so the weights load.
        fc_weight = checkpoint.get("state_dict", {}).get("netD.fc1.weight")
        if fc_weight is not None and (
            self.netD.fc1 is None or self.netD.fc1.in_features != fc_weight.shape[1]
        ):
            self.netD.fc1 = nn.Linear(fc_weight.shape[1], 1)

    def _sample_noise(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.empty(
            batch_size,
            self.config.num_channels,
            self.config.seq_length,
            device=device,
        ).uniform_(-1, 1)

    # ------------------------------------------------------------ conditions

    def _conditions(self, batch: Any):
        """(generator condition, critic condition) for a batch; (None, None) if unconditional."""
        if self.conditioner is None:
            return None, None
        g = self.condition(batch)
        obs = batch.signal if batch.obs_mask is not None else None
        d = self.d_conditioner(
            batch.cond,
            batch.cond_missing,
            obs=obs,
            obs_mask=batch.obs_mask,
            batch_size=len(batch),
        )
        return g, d

    @staticmethod
    def _slots(c) -> tuple[Optional[Tensor], Optional[Tensor]]:
        return (None, None) if c is None else (c.vector, c.dense)

    def _critic(self, x: Tensor, d_cond) -> Tensor:
        vec, dense = self._slots(d_cond)
        return self.netD(x, vec, dense)

    def _generate(self, b: int, device: torch.device, g_cond) -> Tensor:
        vec, dense = self._slots(g_cond)
        return self.netG(self._sample_noise(b, device), vec, dense)

    def _l1(self, fake: Tensor, real: Tensor, batch: Any) -> Tensor:
        obs_mask = getattr(batch, "obs_mask", None)
        if self.config.l1_on == "unobserved" and obs_mask is not None:
            w = 1.0 - obs_mask.to(real.dtype)
            return ((fake - real).abs() * w).sum() / w.sum().clamp_min(1.0)
        return (fake - real).abs().mean()

    # -------------------------------------------------------------- training

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        opt_g, opt_d = self.optimizers()

        real_ecgs = self.get_signal(batch)
        device = real_ecgs.device
        b_size = real_ecgs.size(0)
        g_cond, d_cond = self._conditions(batch)

        # --- Update D ---
        opt_d.zero_grad()
        fake = self._generate(b_size, device, g_cond).detach()

        d_real = self._critic(real_ecgs, d_cond).mean()
        d_fake = self._critic(fake, d_cond).mean()
        gp = calc_gradient_penalty(
            lambda x: self._critic(x, d_cond), real_ecgs, fake, self.config.lmbda
        )

        d_loss = d_fake - d_real + gp
        self.manual_backward(d_loss)
        opt_d.step()

        d_wasserstein = d_real - d_fake

        # --- Update G every n_critic steps ---
        g_loss = torch.tensor(0.0, device=device)
        if (batch_idx + 1) % self.config.n_critic == 0:
            opt_g.zero_grad()
            if g_cond is not None:  # fresh condition dropout for the G step
                g_cond, d_cond = self._conditions(batch)
            fake = self._generate(b_size, device, g_cond)
            g_adv = self._critic(fake, d_cond).mean()
            g_loss = -g_adv
            if self.config.l1_weight > 0:
                l1 = self._l1(fake, real_ecgs, batch)
                g_loss = g_loss + self.config.l1_weight * l1
                self.log("train/l1", l1, on_step=True, on_epoch=True)
            self.manual_backward(g_loss)
            opt_g.step()

        self.log("train/d_loss", d_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/d_wasserstein",
            d_wasserstein,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        self.log("train/g_loss", g_loss, on_step=True, on_epoch=True, prog_bar=True)

        return d_loss

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        real_ecgs = self.get_signal(batch)
        device = real_ecgs.device
        b_size = real_ecgs.size(0)
        g_cond, d_cond = self._conditions(batch)

        fake = self._generate(b_size, device, g_cond)

        d_real = self._critic(real_ecgs, d_cond).mean()
        d_fake = self._critic(fake, d_cond).mean()
        d_wasserstein = d_real - d_fake

        self.log(
            "val/d_wasserstein",
            d_wasserstein,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_loss", -d_wasserstein, on_step=False, on_epoch=True, prog_bar=True
        )
        if g_cond is not None and self.config.l1_weight > 0:
            self.log(
                "val/l1", self._l1(fake, real_ecgs, batch), on_step=False, on_epoch=True
            )

        if batch_idx == 0:
            self._val_real_sample = real_ecgs[0].detach().cpu()
            self._val_fake_sample = fake[0].detach().cpu()

        return d_wasserstein

    # -------------------------------------------------------------- sampling

    @torch.no_grad()
    def generate_samples(self, n_samples: int = 16) -> Tensor:
        """Generate ECG samples from random noise (unconditional models)."""
        self.netG.eval()
        device = next(self.netG.parameters()).device
        noise = self._sample_noise(n_samples, device)
        fake = self.netG(noise)
        return fake

    @torch.no_grad()
    def sample(
        self,
        n_samples: int = 16,
        cond: Optional[dict[str, Any]] = None,
        obs: Optional[Tensor] = None,
        obs_mask: Optional[Tensor] = None,
        paste_observed: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        """
        Generate ECGs; with conditioning, given raw ``cond`` values and/or a
        partially observed ``obs`` (``[n, C, T]``) with ``obs_mask``.
        """
        if self.conditioner is None:
            if cond or obs is not None:
                raise ValueError(
                    "This Pulse2Pulse model was trained without conditioning"
                )
            return self.generate_samples(n_samples)
        self.netG.eval()
        c = self.encode_condition(n_samples, cond, obs, obs_mask)
        fake = self._generate(n_samples, self.device, c)
        if paste_observed and obs is not None:
            m = torch.ones_like(obs) if obs_mask is None else obs_mask
            m, obs = m.to(fake), obs.to(fake)
            fake = obs * m + fake * (1 - m)
        return fake

    def configure_optimizers(self) -> Any:
        g_params = list(self.netG.parameters())
        d_params = list(self.netD.parameters())
        if self.conditioner is not None:
            g_params += list(self.conditioner.parameters())
            d_params += list(self.d_conditioner.parameters())
        opt_g = torch.optim.Adam(
            g_params, lr=self.config.lr, betas=(self.config.b1, self.config.b2)
        )
        opt_d = torch.optim.Adam(
            d_params, lr=self.config.lr, betas=(self.config.b1, self.config.b2)
        )
        return [opt_g, opt_d], []
