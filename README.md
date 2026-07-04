# Do AI Models Collapse Under Recursive Generation of Synthetic Financial Time Series?

Code for a BSc thesis (Econometrics and Operations Research, Erasmus School of
Economics) studying model collapse in financial GANs. Generators are trained
recursively on returns simulated from three stochastic processes with
computable ground truth, and the degradation of tail-risk and bulk
distributional metrics is measured across generations.

## Structure

```
src/
  processes.py             GBM, Merton, and Heston processes with ground-truth VaR and ES
  metrics.py               Tail, bulk, and stylised-fact evaluation metrics
  wgan_gp.py               WGAN-GP generator, critic, and training loop
  quantgan.py              QuantGAN temporal convolutional networks and training loop
  recursion.py             Recursive training procedure (replacement and accumulation)
  gaussian_replication.py  One-dimensional Gaussian baseline experiment
run_experiment.py          Runs the experiments and writes per-generation metrics to results/
analyse_results.py         Reproduces every table and in-text statistic from the CSVs
all_tabels.py              Earlier plain-text summary of the tables
all_plots.py               Generates every figure from the CSVs
results/                   Metric CSVs, one row per generation and seed
figures/                   Generated figures
```

## Requirements

Python 3.10 or later with numpy, scipy, pandas, matplotlib, and torch.

```
pip install numpy scipy pandas matplotlib torch
```

## Reproducing the results

1. Run `python run_experiment.py` to generate the metric CSVs. The active block
   at the top of the file selects the configuration; the commented blocks
   record the settings for the remaining experiments. A full run trains
   several thousand GANs and takes days on a laptop, so the completed CSVs are
   included in `results/`.
2. Run `python analyse_results.py` to print every table and statistic reported
   in the thesis.
3. Run `python all_plots.py` to regenerate the figures.
4. Run `python src/gaussian_replication.py` for the Gaussian baseline figure.

All randomness is controlled by fixed seeds, so the outputs match the values
reported in the thesis exactly.
