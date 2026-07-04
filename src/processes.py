import numpy as np
from scipy import stats


class GBM:
    """Geometric Brownian motion, the light-tailed control process.

    Log returns are independent Gaussian draws, so windows can be sampled
    directly without simulating the price path.
    """

    def __init__(self, mu=0.08, sigma=0.20, dt=1/252, S0=100):
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.S0 = S0
        # Per-step drift and volatility of the log returns.
        self.drift = (mu - 0.5 * sigma**2) * dt
        self.vol = sigma * np.sqrt(dt)

    def sample(self, n_windows, window_length, rng):
        """Draw an array of log-return windows with shape n_windows by window_length."""
        Z = rng.normal(size=(n_windows, window_length))
        log_returns = self.drift + self.vol * Z

        return log_returns

    # The Gaussian distribution gives closed-form risk metrics.
    def true_var(self, alpha=0.01):
        """Analytical Value at Risk at level alpha."""
        z = stats.norm.ppf(alpha)
        var = self.drift + self.vol * z

        return var

    def true_es(self, alpha=0.01):
        """Analytical Expected Shortfall at level alpha."""
        z = stats.norm.ppf(alpha)
        es = self.drift - self.vol * stats.norm.pdf(z) / alpha

        return es


class Merton(GBM):
    """Merton jump diffusion, which adds Poisson jumps to GBM to create heavy tails."""

    def __init__(self, mu=0.08, sigma=0.2, dt=1 / 252, S0=100, lamda=5.0, jump_std=0.045):
        super().__init__(mu, sigma, dt, S0)
        self.lamda = lamda
        self.jump_std = jump_std
        self.reference = None

    def sample(self, n_windows, window_length, rng):
        """Draw log-return windows as the sum of a diffusion and an aggregate jump term.

        The sum of the Gaussian jumps in a step is itself Gaussian with variance
        proportional to the jump count, so the aggregate jump is drawn in one step.
        """
        size = (n_windows, window_length)
        Z = rng.normal(size=size)
        diffusion = self.drift + self.vol * Z

        n_jumps = rng.poisson(lam=self.lamda * self.dt, size=size)
        jumps = self.jump_std * np.sqrt(n_jumps) * rng.normal(size=size)

        log_returns = diffusion + jumps

        return log_returns

    # No closed-form risk metrics exist, so a fixed-seed Monte Carlo reference is used.
    def reference_sample(self):
        """Return a cached reference sample of five million returns drawn with a fixed seed."""
        if self.reference is None:
            rng = np.random.default_rng(11)
            self.reference = self.sample(5000000, 1, rng).ravel()

        return self.reference

    def true_var(self, alpha=0.01):
        """Monte Carlo Value at Risk from the reference sample."""
        var = np.quantile(self.reference_sample(), alpha)

        return var

    def true_es(self, alpha=0.01):
        """Monte Carlo Expected Shortfall from the reference sample."""
        sample = self.reference_sample()
        var = np.quantile(sample, alpha)
        es = np.mean(sample[sample <= var])

        return es


class Heston():
    """Heston stochastic volatility, which adds temporal dependence.

    The variance follows a mean-reverting square-root process, producing the
    volatility clustering that the QuantGAN architecture is designed to capture.
    """

    def __init__(self, mu=0.08, dt=1/252, kappa=0.5, theta=0.08, xi=0.8, rho=-0.7, v0=0.04):
        self.mu = mu
        self.dt = dt
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.v0 = v0
        self.reference = None

    def sample(self, n_windows, window_length, rng):
        """Simulate log-return windows step by step with the full truncation Euler scheme.

        The variance is floored at zero inside the square roots and the drift,
        which handles the violated Feller condition without the bias of simpler
        reflection schemes. Correlated shocks produce the leverage effect.
        """
        returns = np.zeros((n_windows, window_length))
        v = np.full(n_windows, self.v0)

        for t in range(window_length):
            z1 = rng.normal(size=n_windows)
            z2 = rng.normal(size=n_windows)
            zS = z1
            zv = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2

            v_pos = np.maximum(v, 0)
            returns[:, t] = (self.mu - 0.5 * v_pos) * self.dt + np.sqrt(v_pos * self.dt) * zS
            v = v + self.kappa * (self.theta - v_pos) * self.dt + self.xi * np.sqrt(v_pos * self.dt) * zv

        return returns

    def reference_sample(self):
        """Return a cached reference sample of simulated windows drawn with a fixed seed."""
        if self.reference is None:
            rng = np.random.default_rng(123456789)

            self.reference = self.sample(100000, 50, rng).ravel()
        return self.reference

    def true_var(self, alpha=0.01):
        """Monte Carlo Value at Risk from the reference sample."""
        return np.quantile(self.reference_sample(), alpha)

    def true_es(self, alpha=0.01):
        """Monte Carlo Expected Shortfall from the reference sample."""
        sample = self.reference_sample()
        var = np.quantile(sample, alpha)
        return sample[sample <= var].mean()
