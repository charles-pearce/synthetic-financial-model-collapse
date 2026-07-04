import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    """QuantGAN temporal block of two dilated causal convolutions with a residual connection.

    Left padding preserves the sequence length and ensures causality, so each
    output depends only on current and past inputs. A one by one convolution
    aligns the channel count of the residual path when it differs.
    """

    def __init__(self, in_ch, out_ch, kernel_size=2, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation)
        self.act = nn.PReLU()
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = nn.functional.pad(x, (self.pad, 0))
        out = self.act(self.conv1(out))
        out = nn.functional.pad(out, (self.pad, 0))
        out = self.act(self.conv2(out))
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TCN(nn.Module):
    """Temporal convolutional network of stacked blocks with doubling dilations.

    The doubling dilations widen the receptive field exponentially, giving each
    output a dependence on the previous 32 time steps at the default settings.
    """

    def __init__(self, in_ch, out_ch, n_channels=32, dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        layers = []
        c = in_ch
        for d in dilations:
            layers.append(TemporalBlock(c, n_channels, 2, d))
            c = n_channels
        layers.append(nn.Conv1d(c, out_ch, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class QuantGenerator(nn.Module):
    """QuantGAN generator producing a return window from noise supplied at every time step.

    Any temporal structure in the output must be created by the convolutions,
    since the per-step noise vectors are independent.
    """

    def __init__(self, noise_dim=3, seq_len=100):
        super().__init__()
        self.noise_dim = noise_dim
        self.seq_len = seq_len
        self.tcn = TCN(noise_dim, 1)

    def forward(self, z):
        # Input shape is batch by noise dimension by sequence length.
        # The channel dimension is squeezed to return one window per batch element.
        return self.tcn(z).squeeze(1)


class QuantCritic(nn.Module):
    """QuantGAN critic scoring a return window with the same temporal architecture.

    The raw, squared, and absolute returns are stacked as three input channels,
    exposing the variance dynamics directly instead of requiring the critic to
    learn them implicitly.
    """

    def __init__(self, seq_len=100):
        super().__init__()
        self.tcn = TCN(3, 1)

    def forward(self, x):
        features = torch.stack([x, x ** 2, x.abs()], dim=1)
        # Scores are averaged over the time dimension to give one score per window.
        return self.tcn(features).mean(dim=2)


def gradient_penalty(critic, real, fake):
    """Gradient penalty enforcing the Lipschitz constraint on the critic.

    The penalty is computed with respect to the raw return window, before the
    feature stack inside the critic, so the constraint is imposed on the
    function the critic computes from its true input.
    """
    eps = torch.rand(real.size(0), 1, device=real.device)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
    scores = critic(interp)
    grads = torch.autograd.grad(
        outputs=scores, inputs=interp,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
    )[0]
    grad_norm = grads.norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()


def _acf_match_loss(fake, real, lags=(1, 5, 20)):
    """Squared difference between the mean squared-return autocorrelations of the
    generated and real batches, averaged over the given lags.

    The loss is differentiable with respect to the generated batch; the real
    batch should be detached by the caller.
    """
    def batch_mean_acf(x, lag):
        sq = x ** 2
        mu = sq.mean(dim=1, keepdim=True)
        c = sq - mu
        cov = (c[:, :-lag] * c[:, lag:]).mean(dim=1)
        var = (c ** 2).mean(dim=1) + 1e-8
        return (cov / var).mean()

    loss = sum(
        (batch_mean_acf(fake, lag) - batch_mean_acf(real, lag)) ** 2
        for lag in lags
    )
    return loss / len(lags)


def train_quantgan(data, n_iters=3000, batch_size=128, n_critic=5, lam=10,
                   noise_dim=3, seq_len=100, lam_acf=10.0, lazy_gp_every=4):
    """Train a QuantGAN on an array of return windows and return the generator.

    Two departures from the standard WGAN-GP loop make the volatility
    clustering learnable. The critic observes stacked raw, squared, and
    absolute returns, and the generator loss carries an autocorrelation
    matching term. To reduce the cost of second-order gradients, the gradient
    penalty is evaluated only every lazy_gp_every critic steps with its
    coefficient scaled up by the same factor, which leaves the expected
    regularisation unchanged.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    G = QuantGenerator(noise_dim=noise_dim, seq_len=seq_len).to(device)
    C = QuantCritic(seq_len=seq_len).to(device)

    optG = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optC = torch.optim.Adam(C.parameters(), lr=1e-4, betas=(0.5, 0.9))
    data = torch.tensor(data, dtype=torch.float32).to(device)

    scaled_lam = lam * lazy_gp_every

    last_real = None
    critic_step = 0
    for _ in range(n_iters):
        for _ in range(n_critic):
            real = data[torch.randint(0, len(data), (batch_size,))]
            last_real = real
            z = torch.randn(batch_size, noise_dim, seq_len, device=device)
            fake = G(z).detach()
            lossC = C(fake).mean() - C(real).mean()
            if critic_step % lazy_gp_every == 0:
                lossC = lossC + scaled_lam * gradient_penalty(C, real, fake)
            optC.zero_grad(set_to_none=True); lossC.backward(); optC.step()
            critic_step += 1

        z = torch.randn(batch_size, noise_dim, seq_len, device=device)
        fake = G(z)
        lossG = -C(fake).mean() + lam_acf * _acf_match_loss(fake, last_real.detach())
        optG.zero_grad(set_to_none=True); lossG.backward(); optG.step()

    return G
