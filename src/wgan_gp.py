import torch
import torch.nn as nn
import numpy as np
from src.processes import GBM


class Generator(nn.Module):
    """Multilayer perceptron generator mapping a noise vector to a return window."""

    def __init__(self, noise_dim=50, output_length=100):
        super().__init__()
        # The output layer has no activation so the returns are unbounded.
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 128),    nn.LeakyReLU(0.2),
            nn.Linear(128, 128),   nn.LeakyReLU(0.2),
            nn.Linear(128, output_length),
        )

    def forward(self, z):
        return self.net(z)


class Critic(nn.Module):
    """Multilayer perceptron critic mapping a return window to a single realism score."""

    def __init__(self, input_length=100):
        super().__init__()
        # Following the Wasserstein formulation, the output carries no final
        # activation, so it is an unbounded score rather than a probability.
        self.net = nn.Sequential(
            nn.Linear(input_length, 128),  nn.LeakyReLU(0.2),
            nn.Linear(128, 128), nn.LeakyReLU(0.2), 
            nn.Linear(128, 1), 
        )

    def forward(self, x):
        return self.net(x)


def gradient_penalty(critic, real, fake):
    """Gradient penalty enforcing the Lipschitz constraint on the critic.

    The penalty is evaluated at random interpolations between real and
    generated samples and pushes the gradient norm of the critic towards one.
    """
    eps = torch.rand(real.size(0), 1)
    interp = eps * real + (1 - eps) * fake
    interp.requires_grad_(True)

    scores = critic(interp)

    grads = torch.autograd.grad(
        outputs=scores,
        inputs=interp,
        grad_outputs=torch.ones_like(scores), 
        create_graph=True,
    )[0]

    grad_norm = grads.norm(2, dim=1)
    penalty = ((grad_norm - 1)**2).mean()
    return penalty


def train_wgan_gp(data, n_iters=2000, batch_size=128, n_critic=10, lam=10):
    """Train a WGAN-GP on an array of return windows and return the generator.

    Each generator update is preceded by n_critic critic updates. Generated
    samples are detached during the critic updates so that critic gradients
    do not flow into the generator.
    """
    G = Generator()
    C = Critic()
    optG = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optC = torch.optim.Adam(C.parameters(), lr=1e-4, betas=(0.5, 0.9))
    data = torch.tensor(data, dtype=torch.float32)
    
    for _ in range(n_iters):
        for _ in range(n_critic):
            real = data[torch.randint(0, len(data), (batch_size,))]
            fake = G(torch.randn(batch_size, 50)).detach()
            lossC = C(fake).mean() - C(real).mean() + lam * gradient_penalty(C, real, fake)
            optC.zero_grad()
            lossC.backward()
            optC.step()

        fake = G(torch.randn(batch_size, 50))
        lossG = -C(fake).mean()
        optG.zero_grad()
        lossG.backward()
        optG.step()

    return G
