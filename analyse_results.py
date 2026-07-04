"""Reproduces every table and in-text number in the Results section from the CSVs.

Usage: place the result CSVs in ./results/ and run this script. It requires
the src package on the path so that the ground truth can be computed from the
same process classes used to generate the data.

Ground-truth Value at Risk and Expected Shortfall come from each process's own
methods, so they are identical to the values used during data generation.
Kurtosis and the lag-one squared-return autocorrelation have no closed form
and are estimated from each process's reference sample convention, using
windows of length 100 for Heston to match the training window length.
"""
import numpy as np
import pandas as pd
from scipy import stats

# The same classes used to generate the data.
from src.processes import GBM, Merton, Heston


# The CSVs store excess kurtosis while the report prints raw kurtosis, which
# exceeds the excess value by the Gaussian benchmark of three.
KURT_OFFSET = 3.0

# Ground truth for Table 1, computed from the process classes.
def _acf(s, lag):
    return np.corrcoef(s[:-lag], s[lag:])[0, 1]

def ground_truth(proc):
    """Ground-truth risk metrics for one process.

    Value at Risk and Expected Shortfall come from the class methods. Raw
    kurtosis and the lag-one squared-return autocorrelation are estimated from
    the process's fixed-seed reference sample.
    """
    var = proc.true_var(0.01)
    es = proc.true_es(0.01)

    if isinstance(proc, Heston):
        rng = np.random.default_rng(123456789)
        # Windows of length 100 match the training window length.
        windows = proc.sample(100000, 100, rng)
        flat = windows.ravel()
        acf1 = np.nanmean([_acf(w**2, 1) for w in windows])
    else:
        rng = np.random.default_rng(11)
        flat = proc.sample(5000000, 1, rng).ravel()
        # The autocorrelation is zero by construction for the independent processes.
        acf1 = 0.0

    # Raw kurtosis, for which the Gaussian value is three.
    kurt_raw = stats.kurtosis(flat, fisher=False)
    return {"VaR": var, "ES": es, "kurt_raw": kurt_raw, "acf1": acf1}

TRUE = {name: ground_truth(proc)
        for name, proc in [("GBM", GBM()), ("Merton", Merton()), ("Heston", Heston())]}

PROC = {"GBM": "GBM", "Merton": "Merton",
        "Heston QuantGAN": "Heston", "Heston WGAN-GP": "Heston"}

# Result CSVs, one per experimental configuration.
gbm       = pd.read_csv("results/wgangp_gbm_replacement_final.csv")
mer       = pd.read_csv("results/wgangp_merton_replacement_final.csv")
hes_wgan  = pd.read_csv("results/wgangp_heston_replacement_final.csv")
hes_quant = pd.read_csv("results/quantgan_heston_replacement_final.csv")
gacc      = pd.read_csv("results/wgangp_gbm_accumulation_final.csv")
macc      = pd.read_csv("results/wgangp_merton_accumulation_final.csv")
hacc      = pd.read_csv("results/quantgan_heston_accumulation_final.csv")

REPL = [("GBM", gbm), ("Merton", mer),
        ("Heston QuantGAN", hes_quant), ("Heston WGAN-GP", hes_wgan)]


# Helper functions shared by all tables.
def gen0(df, m):
    """Metric values at generation zero, indexed by seed."""
    return df[df.generation == 0].set_index("seed")[m]

def final(df, m):
    """Metric values at the final generation, indexed by seed."""
    last = int(df.generation.max())
    return df[df.generation == last].set_index("seed")[m]

def median_iqr(s):
    """Format a series as its median with the interquartile range in brackets."""
    return f"{s.median():.4f} [{s.quantile(.25):.4f}, {s.quantile(.75):.4f}]"

def paired(a, b):
    """Align two per-seed series on their shared seeds before a paired test."""
    a, b = a.align(b, join="inner")
    return a.values, b.values

def fmt_p(p):
    """Format a p-value, reporting very small values as below one thousandth."""
    return "<0.001" if p < 0.001 else f"{p:.3f}"

def pct(err, true):
    """Express an error as a percentage of the absolute true value."""
    return 100 * err / abs(true)

def rule(char="-", n=94):
    print(char * n)

# Table 1 prints the ground-truth risk metrics per process.
print("=" * 94)
print("TABLE 1 — Process parameters and ground-truth risk metrics")
print("(VaR/ES from each process's true_var()/true_es(); kurtosis and ACF from its reference sample)")
rule()
print(f"{'Process':10} {'True VaR_1%':>12} {'True ES_1%':>12} {'True kurtosis':>14} {'True ACF_r2(1)':>15}")
rule()
for name in ["GBM", "Merton", "Heston"]:
    t = TRUE[name]
    print(f"{name:10} {t['VaR']:>12.4f} {t['ES']:>12.4f} {t['kurt_raw']:>14.2f} {t['acf1']:>15.3f}")
rule()
print("Kurtosis is raw (Gaussian = 3). ACF_r2(1) is the lag-1 autocorrelation of squared returns.")

# Table 2 prints the degradation of each metric under full replacement.
print("\n" + "=" * 94)
print("TABLE 2 — Degradation of metrics under full replacement")
rule()
METRICS = [("var_error", "VaR_1% error"), ("es_error", "ES_1% error"), ("w2", "Wasserstein-2")]
TABLE2_BLOCKS = [("GBM (WGAN-GP)", gbm), ("Merton (WGAN-GP)", mer), ("Heston (QuantGAN)", hes_quant)]

print(f"{'Process (model)':18} {'Metric':13} {'Generation 0':26} {'Final':26} {'Ratio':>6} {'p':>7}")
rule()
for label, df in TABLE2_BLOCKS:
    for i, (col, mlabel) in enumerate(METRICS):
        g0, gT = gen0(df, col), final(df, col)
        p = stats.wilcoxon(gT, g0, alternative="greater").pvalue
        proc = label if i == 0 else ""
        print(f"{proc:18} {mlabel:13} {median_iqr(g0):26} {median_iqr(gT):26} {(gT/g0).median():6.2f} {fmt_p(p):>7}")
    rule()
print("Cells are median across seeds [IQR]. Ratio = median of per-seed (final / generation-0) error.")
print("p = one-sided Wilcoxon signed-rank (H1: final error > generation-0 error).")

# The Heston MLP control appears in the text rather than in Table 2.
print("\n[Heston MLP control — in-text values, Sections 5.2 and 5.5]")
for col, mlabel in METRICS:
    g0, gT = gen0(hes_wgan, col), final(hes_wgan, col)
    p = stats.wilcoxon(gT, g0, alternative="greater").pvalue
    print(f"  {mlabel:13} g0 {median_iqr(g0):26} final {median_iqr(gT):26} ratio {(gT/g0).median():5.2f} p {fmt_p(p)}")

# Table 3 prints the final Value at Risk error under each data condition.
print("\n" + "=" * 94)
print("TABLE 3 — Final VaR_1% error by data condition")
rule()
ACC_BLOCKS = [("GBM (WGAN-GP)", gacc, gbm),
              ("Merton (WGAN-GP)", macc, mer),
              ("Heston (QuantGAN)", hacc, hes_quant)]

print(f"{'Process (model)':18} {'Condition':13} {'Final error':26} {'Ratio':>6} {'p(grow)':>8} {'p(vs repl)':>11}")
rule()
for label, acc, rep in ACC_BLOCKS:
    fr = final(rep, "var_error")
    pg_repl = stats.wilcoxon(fr, gen0(rep, "var_error"), alternative="greater").pvalue
    print(f"{label:18} {'Replacement':13} {median_iqr(fr):26} "
          f"{(fr/gen0(rep,'var_error')).median():6.2f} {fmt_p(pg_repl):>8} {'--':>11}")
    for a in [0.9, 0.75, 0.5]:
        sub = acc[acc.alpha == a]
        g0, fT = gen0(sub, "var_error"), final(sub, "var_error")
        pG = stats.wilcoxon(fT, g0, alternative="greater").pvalue
        fr_a, fT_a = paired(fr, fT)
        pR = stats.wilcoxon(fr_a, fT_a, alternative="greater").pvalue
        cond = f"{int(round((1-a)*100))}% fresh"
        print(f"{'':18} {cond:13} {median_iqr(fT):26} {(fT/g0).median():6.2f} {fmt_p(pG):>8} {fmt_p(pR):>11}")
    rule()
print("Cells are median across seeds [IQR]. Ratio = median per-seed (final / generation-0) error.")
print("p(grow): H1 final > gen-0 within condition.  p(vs repl): paired H1 replacement final > this final.")

# The remaining blocks print every statistic quoted in the running text,
# grouped by the subsection of the Results chapter in which it appears.
print("\n" + "=" * 94)
print("IN-TEXT STATISTICS  (grouped by report subsection)")
print("=" * 94)

# Generation-zero fidelity.
print("\n[5.2 Generation-Zero Fidelity]  kurtosis shown RAW (= CSV excess + 3)")
for name, df in REPL:
    proc = PROC[name]
    v0 = gen0(df, "var_error").median()
    e0 = gen0(df, "es_error").median()
    k0 = gen0(df, "kurtosis").median() + KURT_OFFSET
    a0 = gen0(df, "acf_sq_lag1").median()
    print(f"  {name:16s} VaR err {v0:.4f} ({pct(v0, TRUE[proc]['VaR']):.0f}% of true)  "
          f"ES err {e0:.4f} ({pct(e0, TRUE[proc]['ES']):.0f}%)  "
          f"kurt {k0:.2f} (true {TRUE[proc]['kurt_raw']:.1f})  acf_sq_lag1 {a0:.3f}")

# Collapse under full replacement.
print("\n[5.3 Collapse under full replacement]")
for name, df in [("GBM", gbm), ("Merton", mer), ("Heston QuantGAN", hes_quant)]:
    proc = PROC[name]
    vr = (final(df, "var_error") / gen0(df, "var_error")).median()
    wr = (final(df, "w2") / gen0(df, "w2")).median()
    fin = final(df, "var_error").median()
    med = df.groupby("generation")["var_error"].median()
    rho, _ = stats.spearmanr(med.index, med.values)
    q = df.groupby("generation")["var_error"]
    fan = q.quantile(.75) - q.quantile(.25)
    rv, rw = paired(final(df, "var_error") / gen0(df, "var_error"),
                    final(df, "w2") / gen0(df, "w2"))
    pp = stats.wilcoxon(rv, rw, alternative="greater").pvalue
    print(f"  {name:16s} VaR growth x{vr:.1f}  W2 growth x{wr:.1f}  "
          f"misstatement {pct(fin, TRUE[proc]['VaR']):.0f}% of true VaR  "
          f"Spearman rho={rho:.2f}  IQR fan-out x{fan.iloc[-1]/fan.iloc[0]:.1f}")
    print(f"  {'':16s} primary test (VaR growth > W2 growth): {int((rv>rw).sum())}/{len(rv)} seeds, p={fmt_p(pp)}")

# The role of tail behaviour.
print("\n[5.4 The role of tail behaviour]")
rg = final(gbm, "var_error") / gen0(gbm, "var_error")
rm = final(mer, "var_error") / gen0(mer, "var_error")
print(f"  Mann-Whitney GBM vs Merton VaR growth: p={fmt_p(stats.mannwhitneyu(rg, rm, alternative='two-sided').pvalue)}")
print(f"  Merton kurtosis (raw): g0 {gen0(mer,'kurtosis').median()+KURT_OFFSET:.2f} "
      f"-> final {final(mer,'kurtosis').median()+KURT_OFFSET:.2f}")
print(f"  Merton ES error: g0 {gen0(mer,'es_error').median():.4f} -> final {final(mer,'es_error').median():.4f}")

# Erosion of volatility clustering.
print("\n[5.5 Erosion of volatility clustering]")
lg = int(hes_quant.generation.max())
a0 = gen0(hes_quant, "acf_sq_lag1"); aT = hes_quant[hes_quant.generation == lg].set_index("seed")["acf_sq_lag1"]
a0v, aTv = paired(a0, aT)
print(f"  QuantGAN acf_sq_lag1: g0 {np.median(a0v):.3f} ({pct(np.median(a0v), TRUE['Heston']['acf1']):.0f}% of true) "
      f"-> g{lg} {np.median(aTv):.3f} ({pct(np.median(aTv), TRUE['Heston']['acf1']):.0f}%), "
      f"p={fmt_p(stats.wilcoxon(a0v, aTv, alternative='greater').pvalue)}")
print(f"  QuantGAN acf_sq_lag5 median by gen: "
      f"{hes_quant.groupby('generation')['acf_sq_lag5'].median().round(3).to_dict()}")
print(f"  QuantGAN kurtosis (raw): g0 {gen0(hes_quant,'kurtosis').median()+KURT_OFFSET:.2f} "
      f"-> final {final(hes_quant,'kurtosis').median()+KURT_OFFSET:.2f}")
print(f"  MLP control acf_sq_lag1: g0 {gen0(hes_wgan,'acf_sq_lag1').median():.3f} "
      f"-> g{lg} {hes_wgan[hes_wgan.generation==lg]['acf_sq_lag1'].median():.3f}")

# Reintroducing real data.
print("\n[5.6 Reintroducing real data]")
for label, acc, rep in ACC_BLOCKS:
    fr = final(rep, "var_error").median()
    print(f"  {label}")
    for a in [0.9, 0.75, 0.5]:
        fT = final(acc[acc.alpha == a], "var_error").median()
        print(f"    {int(round((1-a)*100))}% fresh: replacement/accum ratio {fr/fT:.1f}x lower")
print("  Merton kurtosis under accumulation (raw), final by alpha:",
      {a: round(final(macc[macc.alpha==a], "kurtosis").median()+KURT_OFFSET, 2) for a in [0.9,0.75,0.5]})

# Heston clustering under accumulation.
print("\n[Heston QuantGAN clustering under accumulation]  acf_sq_lag1, median across seeds")
for a in [0.9, 0.75, 0.5]:
    sub = hacc[hacc.alpha == a]
    g0 = gen0(sub, "acf_sq_lag1").median()
    fT = final(sub, "acf_sq_lag1").median()
    p = stats.wilcoxon(*paired(gen0(sub, "acf_sq_lag1"),
                               final(sub, "acf_sq_lag1")), alternative="greater").pvalue
    print(f"  {int(round((1-a)*100))}% fresh: g0 {g0:.3f} -> final {fT:.3f} "
          f"({pct(fT, TRUE['Heston']['acf1']):.0f}% of true), p(decline)={fmt_p(p)}")
