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

# The commented functions below are earlier exploratory versions retained for reference.

# def drift_variance_theory(n_samples, n_generations, sigma0=1.0):
#     k = np.arange(n_generations + 1)
#     return k * sigma0**2 / n_samples

# def plot_mu_sigma_by_sample_size(sample_sizes, n_generations, mu0=0.0, sigma0=1.0):
#     """One chain per sample size; plot absolute error in mu and sigma over generations."""
#     fig, (ax_mu, ax_sigma) = plt.subplots(1, 2, figsize=(14, 4.8))

#     for n in sample_sizes:
#         rng = np.random.default_rng(n)
#         hist = gaussian_recursion(n, n_generations, rng, mu0, sigma0)
#         gens = np.arange(len(hist))
#         ax_mu.plot(gens, np.abs(hist[:, 0] - mu0), label=str(n), lw=1)
#         ax_sigma.plot(gens, np.abs(hist[:, 1] - sigma0), label=str(n), lw=1)

#     ax_mu.set(title=r"$\hat\mu$ estimation of a $\mathcal{N}(\mu=0,\sigma=1)$",
#               xlabel="evolution", ylabel=r"$|\mu - \hat\mu|$", xscale="log")
#     ax_sigma.set(title=r"$\hat\sigma$ estimation of a $\mathcal{N}(\mu=0,\sigma=1)$",
#                  xlabel="evolution", ylabel=r"$|\sigma - \hat\sigma|$", xscale="log")
#     ax_mu.legend(); ax_sigma.legend()
#     fig.tight_layout()
#     return fig


# def plot_tail_histogram(M, n_bins=25, rng=None):
#     """Show two distributions that differ in their tails collapsing to one after resampling."""
#     if rng is None:
#         rng = np.random.default_rng(0)

#     def bimodal(spread_fill):
#         left = rng.normal(-4, 1.0, M // 2)
#         right = rng.normal(4, 1.0, M // 2)
#         extra = rng.normal(0, spread_fill, M // 10)
#         return np.concatenate([left, right, extra])

#     bins = np.linspace(-10, 10, n_bins)
#     centres = 0.5 * (bins[:-1] + bins[1:])
#     real1, real2 = bimodal(3.0), bimodal(1.2)

#     probs = np.histogram(real1, bins)[0]
#     probs = probs / probs.sum()
#     resampled = centres[rng.choice(len(centres), size=M, p=probs)]

#     logM = np.log(M) - np.log(len(centres))
#     fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
#     for ax, data, title in [(axes[0], real1, "Real distribution 1"),
#                             (axes[1], real2, "Real distribution 2"),
#                             (axes[2], resampled, "Resampled 1 and 2")]:
#         c = np.histogram(data, bins)[0]
#         logc = np.log(np.where(c > 0, c, np.nan))
#         ax.bar(centres, logc, width=(bins[1]-bins[0])*0.9, color="y", edgecolor="k")
#         ax.axhline(logM, color="red", lw=2, label="log M")
#         ax.set(title=title, xlim=(-10, 10), ylim=(0, 12))
#         ax.grid(alpha=0.3)
#     axes[0].set_ylabel("log(Count)")
#     axes[2].legend()
#     fig.tight_layout()
#     return fig

# fig = plot_mu_sigma_by_sample_size([100, 500, 1000, 10000], n_generations=1000)
# fig.savefig("mu_sigma.pdf")

def plot_gaussian_replication(fname="gaussian_replication_sampling2.pdf",
                              sample_sizes=(100, 500, 1000, 10000),
                              n_generations=100, n_chains=3000, seed=1):
    """Three-panel baseline figure showing the drift of the mean, the collapse of
    the standard deviation, and the tail against bulk error."""
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9.5, "axes.titlesize": 10,
        "axes.labelsize": 9.5, "legend.fontsize": 8.5, "figure.dpi": 150,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    })
    rng = np.random.default_rng(seed)
    # Palette order is blue, orange, green, red, purple.
    palette = ["#2077B4", "#FF7F15", "#00A170", "#4F130C", "#BE8FEF"]
    colors = {n: palette[i] for i, n in enumerate(sample_sizes)}

    def fmt(n):
        e = int(round(np.log10(n)))
        return rf"$n=10^{{{e}}}$" if n == 10 ** e else rf"$n={n:,}$".replace(",", "{,}")

    # True one percent Value at Risk of the standard normal, its alpha quantile.
    z_alpha = stats.norm.ppf(0.01)

    fig, (ax_drift, ax_sigma, ax_tail) = plt.subplots(1, 3, figsize=(15.5, 4.4))
    gens = np.arange(n_generations + 1)

    for n in sample_sizes:
        mu_hist, sigma_hist = run_many_chains(n, n_generations, n_chains, rng)
        c = colors[n]
        # Panel 1 shows the mean absolute error in the estimated mean across
        # chains, with the theoretical prediction overlaid as a dashed line.
        ax_drift.plot(gens, np.abs(mu_hist).mean(axis=0), color=c)
        ax_drift.plot(gens, np.sqrt(2 / np.pi) * np.sqrt(drift_variance_theory(n, n_generations)),
                      color=c, ls="--", lw=1, alpha=0.55)
        # Panel 2 shows the mean absolute error in the estimated standard deviation.
        ax_sigma.plot(gens, np.abs(1 - sigma_hist).mean(axis=0), color=c)
        # Panel 3 compares the tail and bulk errors on the same convention.
        # The bulk error is the absolute error in the centre and the tail error
        # is the absolute error in the one percent Value at Risk. The tail
        # inherits the systematic bias of the collapsing standard deviation
        # that the bulk does not, so it degrades faster. Solid lines show the
        # tail and dashed lines show the bulk.
        bulk = np.abs(mu_hist).mean(axis=0)
        tail = np.abs(mu_hist + sigma_hist * z_alpha - z_alpha).mean(axis=0)
        ax_tail.plot(gens, tail, color=c)
        ax_tail.plot(gens, bulk, color=c, ls="--", lw=1.1)

    ax_drift.set(xlabel="Generation", ylabel=r"$|\mu - \hat\mu|$ across chains",
                 title="Drift of the estimated mean")
    ax_sigma.axhline(0, color="grey", lw=0.8, ls=":")
    ax_sigma.set(xlabel="Generation", ylabel=r"$|\sigma - \hat\sigma|$ across chains",
                 title="Collapse of the estimated standard deviation")
    ax_sigma.axhline(0, color="grey", lw=0.8, ls=":")
    ax_tail.set(xlabel="Generation", ylabel="Mean absolute error vs true value",
                title="Tail vs bulk error", yscale="log")

    handles = [Line2D([0], [0], color=colors[n], label=fmt(n)) for n in sample_sizes]
    handles.append(Line2D([0], [0], color="grey", ls="--",
                          label=r"theory $\sqrt{2/\pi}\,\sqrt{k}\,\sigma_0/\sqrt{n}$"))
    ax_drift.legend(handles=handles, fontsize=7.5, loc="upper left")
    ax_sigma.legend(handles=handles[:-1], fontsize=7.5, loc="lower left")
    style_handles = [Line2D([0], [0], color="k", label="1% VaR (tail)"),
                     Line2D([0], [0], color="k", ls="--", label="mean (bulk)")]
    ax_tail.legend(handles=style_handles, fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)
    return fig

if __name__ == "__main__":
    plot_gaussian_replication()
