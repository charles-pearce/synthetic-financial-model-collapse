import numpy as np
import torch
from src.wgan_gp import train_wgan_gp
from src.metrics import all_metrics


def run_recursion(process, condition, alpha, n_generations, n_iters, seed,
                  train_fn=train_wgan_gp, noise_shape=(5000, 50)):
    """Run one recursive training chain and return per-generation metric records.

    A generation is one cycle of training a model on the current dataset and
    sampling it to produce the next dataset. Under the replacement condition
    the next dataset is entirely synthetic. Under the accumulation condition a
    fraction alpha of the windows is synthetic and the remainder is freshly
    simulated from the true process at every generation, with the dataset size
    held fixed. The torch seed is varied per generation so that each training
    run is distinct but reproducible.

    process is a data-generating process from src.processes, condition is
    replacement or accumulation, alpha is the synthetic fraction, train_fn is
    the training routine, and noise_shape is the sampling noise shape expected
    by the trained generator. Returns a list of metric dictionaries, one per
    generation.
    """
    rng = np.random.default_rng(seed)
    reference = process.sample(50000, 100, rng).ravel()
    data = process.sample(5000, 100, rng)

    records = []
    for gen in range(n_generations):
        torch.manual_seed(seed * 1000 + gen)
        G = train_fn(data, n_iters=n_iters)

        with torch.no_grad():
            device = next(G.parameters()).device
            synthetic = G(torch.randn(*noise_shape, device=device)).cpu().numpy()

        m = all_metrics(synthetic, process, reference)
        m["generation"] = gen
        m["seed"] = seed
        m["condition"] = condition
        m["alpha"] = alpha
        records.append(m)

        if condition == "replacement":
            data = synthetic
        elif condition == "accumulation":
            n_real = int((1 - alpha) * 5000)
            fresh = process.sample(n_real, 100, rng)
            data = np.concatenate([synthetic[:5000 - n_real], fresh])

    return records
