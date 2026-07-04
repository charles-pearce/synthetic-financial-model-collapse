"""Generates every figure in the thesis from the result CSV files.

Figures are written to the figures directory. Each plotting function skips
silently when its input CSV is absent, so the script can be run on a partial
set of results.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from src.processes import GBM, Merton, Heston
from src.gaussian_replication import run_many_chains, drift_variance_theory
from scipy import stats

rng = np.random.default_rng(0)

# Consistent figure style applied to every plot in the thesis.
mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "font.family": "serif",
    "font.serif": "Times New Roman",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 5,
    "legend.frameon": False,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "mathtext.default": "regular",
    "legend.fontsize": 10,  
})

blue = "#2077B4"  
orange = "#FF7F15"   
green = "#00A170"   
red = "#4F130C"
purple = "#BE8FEF"

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
RES = "results"   

def load(name):
    """Load a results CSV by name, returning None when the file does not exist."""
    path = os.path.join(RES, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def plot_collapse(df, title, fname):
    """Tail and bulk error against generation, median with an interquartile band across seeds."""
    if df is None:
        print(f"  skip {fname} (no data)"); return
    gens = sorted(df.generation.unique())
    fig, ax = plt.subplots(figsize=(4, 2.7))
    for metric, colour, label in [
        ("var_error", blue, "1% VaR error (tail)"),
        ("es_error",  orange,  "1% ES error (tail)"),
        ("w2",        green,     "Wasserstein-2 (bulk)"),
    ]:
        grouped = df.groupby("generation")[metric]
        median = grouped.median()
        lo = grouped.quantile(0.25)
        hi = grouped.quantile(0.75)
        ax.fill_between(gens, lo, hi, color=colour, alpha=0.15, linewidth=0)
        ax.plot(gens, median, color=colour, label=label)
    ax.set_xticks(range(int(min(gens)), int(max(gens)) + 1, 2))
    ax.set_xlabel("Generation")
    ax.set_ylabel("Error relative to true distribution")
    ax.set_title(title)
    # Extra headroom keeps the legend clear of the rising curves.
    ax.set_ylim(top=ax.get_ylim()[1] * 1.25)
    ax.legend()
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print(f"  saved {fname}")


def plot_accumulation(df, title, fname):
    """Value at Risk error against generation, one line per mixing ratio, median with
    an interquartile band across seeds."""
    if df is None:
        print(f"  skip {fname} (no data)"); return
    gens = sorted(df.generation.unique())
    fig, ax = plt.subplots(figsize=(4, 2.7))
    colours = [blue, orange, green]
    for a, colour in zip(sorted(df.alpha.unique(), reverse=True), colours):
        sub = df[df.alpha == a]
        grouped = sub.groupby("generation")["var_error"]
        median = grouped.median()
        lo = grouped.quantile(0.25)
        hi = grouped.quantile(0.75)
        pct_real = int((1 - a) * 100)
        ax.fill_between(gens, lo, hi, color=colour, alpha=0.15, linewidth=0)
        ax.plot(gens, median, color=colour, label=f"alpha={a}")
    ax.set_xticks(range(int(min(gens)), int(max(gens)) + 1, 2))
    ax.set_xlabel("Generation")
    ax.set_ylabel("1% VaR error")
    ax.set_title(title)
    ax.legend()
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print(f"  saved {fname}")

def plot_accumulation2(df, title, fname, repl_df=None):
    """Accumulation figure including the full replacement condition for comparison.

    Pass the matching replacement dataframe as repl_df to draw it as the line
    with alpha equal to one.
    """
    if df is None:
        print(f"  skip {fname} (no data)"); return
    gens = sorted(df.generation.unique())
    fig, ax = plt.subplots(figsize=(4, 2.7))

    # The replacement line is drawn first so it sits behind the accumulation lines.
    if repl_df is not None:
        grouped = repl_df.groupby("generation")["var_error"]
        median = grouped.median()
        lo = grouped.quantile(0.25)
        hi = grouped.quantile(0.75)
        ax.fill_between(gens, lo, hi, color=red, alpha=0.15, linewidth=0)
        ax.plot(gens, median, color=red, label="alpha=1")

    colours = [blue, orange, green]
    for a, colour in zip(sorted(df.alpha.unique(), reverse=True), colours):
        sub = df[df.alpha == a]
        grouped = sub.groupby("generation")["var_error"]
        median = grouped.median()
        lo = grouped.quantile(0.25)
        hi = grouped.quantile(0.75)
        pct_real = int((1 - a) * 100)
        ax.fill_between(gens, lo, hi, color=colour, alpha=0.15, linewidth=0)
        ax.plot(gens, median, color=colour, label=f"alpha={a}")

    ax.set_xticks(range(int(min(gens)), int(max(gens)) + 1, 2))
    ax.set_xlabel("Generation")
    ax.set_ylabel("1% VaR error")
    ax.set_title(title)
    ax.legend()
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print(f"  saved {fname}")


def plot_clustering(df, title, fname):
    """Squared-return autocorrelation against generation for the Heston process,
    median with an interquartile band across seeds."""
    if df is None:
        print(f"  skip {fname} (no data)"); return
    gens = sorted(df.generation.unique())
    fig, ax = plt.subplots(figsize=(4.7, 3.17))
    for metric, colour, label in [
        ("acf_sq_lag1", blue, "Lag 1"),
        ("acf_sq_lag5", orange,  "Lag 5"),
    ]:
        grouped = df.groupby("generation")[metric]
        median = grouped.median()
        lo = grouped.quantile(0.25)
        hi = grouped.quantile(0.75)
        ax.fill_between(gens, lo, hi, color=colour, alpha=0.15, linewidth=0)
        ax.plot(gens, median, color=colour, label=label)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Squared-return autocorrelation")
    ax.set_title(title)
    ax.legend()
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print(f"  saved {fname}")

def plot_accumulation_acf(df, title, fname, true_acf=0.17):
    """Lag-one squared-return autocorrelation against generation, one line per
    mixing ratio, median with an interquartile band across seeds. A dashed line
    marks the true process value."""
    if df is None:
        print(f"  skip {fname} (no data)"); return
    gens = sorted(df.generation.unique())
    fig, ax = plt.subplots(figsize=(4.7, 3.17))
    colours = [blue, orange, green]
    for a, colour in zip(sorted(df.alpha.unique(), reverse=True), colours):
        sub = df[df.alpha == a]
        grouped = sub.groupby("generation")["acf_sq_lag1"]
        median = grouped.median()
        lo = grouped.quantile(0.25)
        hi = grouped.quantile(0.75)
        pct_real = int((1 - a) * 100)
        ax.fill_between(gens, lo, hi, color=colour, alpha=0.15, linewidth=0)
        ax.plot(gens, median, color=colour, label=f"alpha={a}")
    ax.axhline(true_acf, color="black", ls="--", lw=1, label="true process")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Squared-return autocorrelation (lag 1)")
    ax.set_title(title)
    ax.legend()
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print(f"  saved {fname}")
 
def plot_example_returns(fname="example_returns.pdf", seed=1, length=100):
    """One example log-return window from each process, stacked in three panels."""
    rng = np.random.default_rng(seed)
    series = [
        ("GBM",    GBM().sample(1, length, rng)[0]),
        ("Merton", Merton().sample(1, length, rng)[0]),
        ("Heston", Heston().sample(1, length, rng)[0]),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(4.7, 3.17), sharex=True)
    for ax, (name, r) in zip(axes, series):
        ax.plot(r, color=blue)
        ax.set_ylabel(f"{name}\nlog return")
        ax.axhline(0, color="k", lw=0.5)
    axes[-1].set_xlabel("Trading day")
    fig.suptitle("Example log returns")
    fig.savefig(os.path.join(OUT, fname)); plt.close(fig)
    print(f"  saved {fname}")

def plot_heston_acf(data, title, fname, max_lag=25):
    """Autocorrelation of raw and squared returns by lag, averaged over windows.

    Shows the stylised facts of the simulated Heston process, with clustering
    visible in the squared returns and none in the raw returns.
    """
    if data is None or len(data) == 0:
        print(f"  skip {fname} (no data)"); return

    def acf(s, lag):
        return np.corrcoef(s[:-lag], s[lag:])[0, 1]

    lags = range(1, max_lag + 1)
    raw_acf = [np.nanmean([acf(w, l) for w in data]) for l in lags]
    sq_acf = [np.nanmean([acf(w ** 2, l) for w in data]) for l in lags]

    fig, ax = plt.subplots(figsize=(4.7, 3.17))
    ax.bar([l - 0.2 for l in lags], raw_acf, width=0.4, color=blue, label="raw returns")
    ax.bar([l + 0.2 for l in lags], sq_acf, width=0.4, color=orange, label="squared returns")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(title)
    ax.legend()
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print(f"  saved {fname}")
 
# One-dimensional Gaussian replication of the baseline collapse experiment.
 
def plot_gaussian_replication(fname="gaussian_replication"):
    """Save the three Gaussian baseline panels as separate PDF files.

    The chains are simulated once and reused across the drift, standard
    deviation, and tail against bulk panels.
    """
    rng = np.random.default_rng(1)
    n_gen, n_chains = 1000, 3000
    sample_sizes = [100, 1000, 10000]  

    results = {}
    for n in sample_sizes:
        mu_hist, sigma_hist = run_many_chains(n, n_gen, n_chains, rng)
        results[n] = (mu_hist, sigma_hist)

    colours = {100: blue, 1000: orange, 10000: green}

    def fmt(n):
        exp = int(np.log10(n))
        return f"$n = 10^{{{exp}}}$" if n == 10 ** exp else f"$n = {n}$"

    from matplotlib.lines import Line2D
    z = stats.norm.ppf(0.01)
    gens = np.arange(n_gen + 1)

    def save_panel(suffix, draw, xlabel, ylabel, title, legend_handles, yscale=None):
        fig, ax = plt.subplots(figsize=(4, 2.7))
        draw(ax)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        if yscale:
            ax.set_yscale(yscale)
        # ax.set_ylim(top=ax.get_ylim()[1] * 1.25)
        if legend_handles:
            ax.legend(handles=legend_handles)
        fig.tight_layout()
        out = os.path.join(OUT, f"{fname}_{suffix}.pdf")
        fig.savefig(out)
        plt.close(fig)
        print(f"  saved {fname}_{suffix}.pdf")

    # First panel shows the root mean square drift of the estimated mean with
    # the theoretical prediction overlaid.
    def draw_drift(ax):
        for n in sample_sizes:
            mu_hist, _ = results[n]
            ax.plot(gens, np.sqrt(mu_hist.var(axis=0)), color=colours[n])
            ax.plot(gens, np.sqrt(drift_variance_theory(n, n_gen)),
                    color=colours[n], ls="--", lw=1, alpha=0.55)
    drift_handles = [Line2D([0], [0], color=colours[n], label=fmt(n)) for n in sample_sizes]
    drift_handles.append(Line2D([0], [0], color="grey", ls="--",
                                label=r"theory"))
    save_panel("drift", draw_drift, "Generation", r"RMS drift of $\hat\mu$ across chains",
               "Drift of the estimated mean", drift_handles)

    # Second panel shows the mean of the estimated standard deviation across
    # chains, which collapses towards zero.
    def draw_sigma(ax):
        for n in sample_sizes:
            _, sigma_hist = results[n]
            ax.plot(gens, sigma_hist.mean(axis=0), color=colours[n])
    sigma_handles = [Line2D([0], [0], color=colours[n], label=fmt(n)) for n in sample_sizes]
    save_panel("sigma", draw_sigma, "Generation", r"Mean of $\hat\sigma$ across chains",
               "Collapse of the estimated standard deviation", sigma_handles)

    # Third panel compares the error in the one percent Value at Risk with the
    # error in the mean on a logarithmic scale. Solid lines show the tail and
    # dashed lines show the bulk.
    def draw_tail(ax):
        for n in sample_sizes:
            mu_hist, sigma_hist = results[n]
            bulk = np.sqrt((mu_hist ** 2).mean(axis=0))
            tail = np.sqrt(((mu_hist + sigma_hist * z - z) ** 2).mean(axis=0))
            ax.plot(gens, tail, color=colours[n])
            ax.plot(gens, bulk, color=colours[n], ls="--", lw=1.1)
    tail_handles = [Line2D([0], [0], color="k", label="1% VaR (tail)"),
                    Line2D([0], [0], color="k", ls="--", label="mean (bulk)")]
    save_panel("tail", draw_tail, "Generation", "RMS error vs true value",
               "Tail vs bulk error", tail_handles, yscale="log")

print("Generating figures:")

# The active calls below generate the figures used in the thesis. The
# commented calls generate alternative versions and can be reactivated.

plot_example_returns()
# plot_gaussian_replication()

plot_collapse(load("wgangp_gbm_replacement_final.csv"),    "WGAN-GP, GBM, replacement",    "collapse_gbm_final3.pdf")
plot_collapse(load("wgangp_merton_replacement_final.csv"), "WGAN-GP, Merton, replacement", "collapse_merton_final3.pdf")
plot_collapse(load("wgangp_heston_replacement_final.csv"),    "WGAN-GP, Heston, replacement",    "collapse_heston_wgangp_final3.pdf")
plot_collapse(load("quantgan_heston_replacement_final.csv"), "QuantGAN, Heston, replacement", "collapse_heston_final3.pdf")

# plot_accumulation(load("wgangp_gbm_accumulation_final.csv"),    "WGAN-GP, GBM, accumulation",    "accumulation_gbm_final3.pdf")
# plot_accumulation(load("wgangp_merton_accumulation_final.csv"), "WGAN-GP, Merton, accumulation", "accumulation_merton_final3.pdf")
# plot_accumulation(load("quantgan_heston_accumulation_final.csv"), "QuantGAN, Heston, accumulation", "accumulation_heston_final3.pdf")

plot_accumulation2(load("wgangp_gbm_accumulation_final.csv"),   "WGAN-GP, GBM, accumulation",      "accumulation_gbm1.pdf",    repl_df=load("wgangp_gbm_replacement_final.csv"))
plot_accumulation2(load("wgangp_merton_accumulation_final.csv"),   "WGAN-GP, Merton, accumulation",   "accumulation_merton1.pdf", repl_df=load("wgangp_merton_replacement_final.csv"))
plot_accumulation2(load("quantgan_heston_accumulation_final.csv"),   "QuantGAN, Heston, accumulation",  "accumulation_heston1.pdf", repl_df=load("quantgan_heston_replacement_final.csv"))

# plot_clustering(load("quantgan_heston_replacement_final.csv"), "QuantGAN, Heston, volatility clustering", "clustering_heston_final3.pdf")
# plot_accumulation_acf(load("quantgan_heston_accumulation_final.csv"), "QuantGAN, Heston, accumulation", "accumulation_clustering_heston_final3.pdf")
# heston = Heston()
# plot_heston_acf(heston.sample(5000, 100, rng), "Heston stylised facts", "heston_acf_final3.pdf")

print("Done.")
