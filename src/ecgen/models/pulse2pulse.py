from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import pytorch_lightning as pl


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
        self.upsample_layer = nn.Upsample(scale_factor=upsample) if upsample is not None else None
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
        self.upsample_layer = nn.Upsample(scale_factor=upsample) if upsample is not None else None
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
    ) -> None:
        super().__init__()
        self.model_size = model_size
        self.num_channels = num_channels
        self.post_proc_filt_len = post_proc_filt_len
        self.verbose = verbose

        stride = 1 if upsample else 4
        upsample_val: Optional[int] = 5 if upsample else None

        self.deconv_1 = Transpose1dLayer(5 * model_size, 5 * model_size, 25, stride, upsample=upsample_val)
        self.deconv_2 = Transpose1dLayerMultiInput(5 * model_size * 2, 3 * model_size, 25, stride, upsample=upsample_val)
        self.deconv_3 = Transpose1dLayerMultiInput(3 * model_size * 2, model_size, 25, stride, upsample=upsample_val)
        self.deconv_5 = Transpose1dLayerMultiInput(model_size * 2, model_size // 2, 25, stride, upsample=2)
        self.deconv_6 = Transpose1dLayerMultiInput(model_size, model_size // 5, 25, stride, upsample=upsample_val)
        self.deconv_7 = Transpose1dLayer(model_size // 5, num_channels, 25, stride, upsample=2)

        self.conv_1 = nn.Conv1d(num_channels, model_size // 5, 25, stride=2, padding=25 // 2)
        self.conv_2 = nn.Conv1d(model_size // 5, model_size // 2, 25, stride=5, padding=25 // 2)
        self.conv_3 = nn.Conv1d(model_size // 2, model_size, 25, stride=2, padding=25 // 2)
        self.conv_4 = nn.Conv1d(model_size, model_size * 3, 25, stride=5, padding=25 // 2)
        self.conv_5 = nn.Conv1d(model_size * 3, model_size * 5, 25, stride=5, padding=25 // 2)
        self.conv_6 = nn.Conv1d(model_size * 5, model_size * 5, 25, stride=5, padding=25 // 2)

        if post_proc_filt_len and post_proc_filt_len > 0:
            self.post_proc_filter = nn.Conv1d(num_channels, num_channels, post_proc_filt_len)
        else:
            self.post_proc_filter = None

        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x: Tensor) -> Tensor:
        conv_1_out = F.leaky_relu(self.conv_1(x))
        conv_2_out = F.leaky_relu(self.conv_2(conv_1_out))
        conv_3_out = F.leaky_relu(self.conv_3(conv_2_out))
        conv_4_out = F.leaky_relu(self.conv_4(conv_3_out))
        conv_5_out = F.leaky_relu(self.conv_5(conv_4_out))
        x = F.leaky_relu(self.conv_6(conv_5_out))

        x = F.relu(self.deconv_1(x))
        x = F.relu(self.deconv_2(x, conv_5_out))
        x = F.relu(self.deconv_3(x, conv_4_out))
        x = F.relu(self.deconv_5(x, conv_3_out))
        x = F.relu(self.deconv_6(x, conv_2_out))
        x = torch.tanh(self.deconv_7(x))

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

        k_list = torch.randint(
            low=0,
            high=2 * self.shift_factor + 1,
            size=(x.shape[0],),
            device=x.device,
        ) - self.shift_factor

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
    ) -> None:
        super().__init__()
        self.model_size = model_size
        self.num_channels = num_channels
        self.alpha = alpha
        self.verbose = verbose

        self.conv1 = nn.Conv1d(num_channels, model_size, 25, stride=2, padding=11)
        self.conv2 = nn.Conv1d(model_size, 2 * model_size, 25, stride=2, padding=11)
        self.conv3 = nn.Conv1d(2 * model_size, 5 * model_size, 25, stride=2, padding=11)
        self.conv4 = nn.Conv1d(5 * model_size, 10 * model_size, 25, stride=2, padding=11)
        self.conv5 = nn.Conv1d(10 * model_size, 20 * model_size, 25, stride=4, padding=11)
        self.conv6 = nn.Conv1d(20 * model_size, 25 * model_size, 25, stride=4, padding=11)
        self.conv7 = nn.Conv1d(25 * model_size, 100 * model_size, 25, stride=4, padding=11)

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

    def _ensure_fc(self, x: Tensor) -> None:
        flat_dim = x.size(1) * x.size(2)
        if self.fc1 is None or getattr(self.fc1, "in_features", None) != flat_dim:
            self.fc1 = nn.Linear(flat_dim, 1, device=x.device)
            nn.init.kaiming_normal_(self.fc1.weight.data)

    def forward(self, x: Tensor) -> Tensor:
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

        x = x.view(x.size(0), -1, 1)
        self._ensure_fc(x)
        assert self.fc1 is not None
        x = x.view(x.size(0), -1)
        return self.fc1(x)


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


class Pulse2PulseGAN(pl.LightningModule):
    """
    PyTorch Lightning wrapper around the Pulse2Pulse WaveGAN generator + discriminator.
    """

    def __init__(self, config: Pulse2PulseConfig | dict[str, Any]) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = Pulse2PulseConfig(**config)
        self.save_hyperparameters(config.__dict__)
        self.config = config

        self.netG = WaveGANGenerator(
            model_size=config.model_size,
            num_channels=config.num_channels,
        )
        self.netD = WaveGANDiscriminator(
            model_size=config.model_size,
            num_channels=config.num_channels,
        )

        self.automatic_optimization = False
        
        self._val_real_sample: Tensor | None = None
        self._val_fake_sample: Tensor | None = None

    def forward(self, noise: Tensor) -> Tensor:
        return self.netG(noise)

    def _sample_noise(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.empty(
            batch_size,
            self.config.num_channels,
            self.config.seq_length,
            device=device,
        ).uniform_(-1, 1)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        opt_g, opt_d = self.optimizers()

        real_ecgs: Tensor = batch["ecg_signals"]
        device = real_ecgs.device
        b_size = real_ecgs.size(0)

        # --- Update D ---
        opt_d.zero_grad()
        noise = self._sample_noise(b_size, device)
        fake = self.netG(noise).detach()

        d_real = self.netD(real_ecgs).mean()
        d_fake = self.netD(fake).mean()
        gp = calc_gradient_penalty(self.netD, real_ecgs, fake, self.config.lmbda)

        d_loss = d_fake - d_real + gp
        self.manual_backward(d_loss)
        opt_d.step()

        d_wasserstein = d_real - d_fake

        # --- Update G every n_critic steps ---
        g_loss = torch.tensor(0.0, device=device)
        if (batch_idx + 1) % self.config.n_critic == 0:
            opt_g.zero_grad()
            noise = self._sample_noise(b_size, device)
            fake = self.netG(noise)
            g_adv = self.netD(fake).mean()
            g_loss = -g_adv
            self.manual_backward(g_loss)
            opt_g.step()

        self.log("train/d_loss", d_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/d_wasserstein", d_wasserstein, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/g_loss", g_loss, on_step=True, on_epoch=True, prog_bar=True)

        return d_loss

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        real_ecgs: Tensor = batch["ecg_signals"]
        device = real_ecgs.device
        b_size = real_ecgs.size(0)

        noise = self._sample_noise(b_size, device)
        fake = self.netG(noise)

        d_real = self.netD(real_ecgs).mean()
        d_fake = self.netD(fake).mean()
        d_wasserstein = d_real - d_fake

        self.log("val/d_wasserstein", d_wasserstein, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss", -d_wasserstein, on_step=False, on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            self._val_real_sample = real_ecgs[0].detach().cpu()
            self._val_fake_sample = fake[0].detach().cpu()

        return d_wasserstein

    @torch.no_grad()
    def generate_samples(self, n_samples: int = 16) -> Tensor:
        """Generate ECG samples from random noise."""
        self.netG.eval()
        device = next(self.netG.parameters()).device
        noise = self._sample_noise(n_samples, device)
        fake = self.netG(noise)
        return fake

    def configure_optimizers(self) -> Any:
        opt_g = torch.optim.Adam(
            self.netG.parameters(),
            lr=self.config.lr,
            betas=(self.config.b1, self.config.b2),
        )
        opt_d = torch.optim.Adam(
            self.netD.parameters(),
            lr=self.config.lr,
            betas=(self.config.b1, self.config.b2),
        )
        return [opt_g, opt_d], []

