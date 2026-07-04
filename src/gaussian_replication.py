import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats


def gaussian_recursion(n_samples, n_generations, rng, mu0=0.0, sigma0=1.0):
    """Run one recursive Gaussian chain and return the parameter history.

    Each generation draws a sample from the current Gaussian fit and refits the
    mean and standard deviation, so estimation error compounds across
    generations. This replicates the one-dimensional baseline experiment of
    Shumailov and coauthors.
    """
    mu = mu0
    sigma = sigma0
    history = [(mu, sigma)]
    
    for _ in range(n_generations):
        sample = rng.normal(mu, sigma, size=n_samples)
        mu = sample.mean()
        sigma = sample.std(ddof=1)
        history.append((mu, sigma))

    return np.array(history)

def run_many_chains(n_samples, n_generations, n_chains, rng, mu=0.0, sigma=1.0):
    """Run independent chains and return the mean and standard deviation histories."""
    results = [gaussian_recursion(n_samples, n_generations, rng, mu, sigma)
               for _ in range(n_chains)]
    results = np.array(results)
    return results[:, :, 0], results[:, :, 1]

def drift_variance_theory(n_samples, n_generations, sigma0=1.0):
    """Theoretical variance of the drifting mean, growing linearly in the generation index."""
    k = np.arange(n_generations + 1)
    return k * sigma0**2 / n_samples