import numpy as np
from scipy import stats
from src import processes


# Tail metrics
def empirical_var(returns, alpha=0.01):
    """Empirical Value at Risk, the alpha quantile of the returns."""
    return np.quantile(returns, alpha)

def empirical_es(returns, alpha=0.01):
    """Empirical Expected Shortfall, the mean of the returns beyond the Value at Risk."""
    var = np.quantile(returns, alpha)
    return np.mean(returns[returns <= var])

def kurtosis(returns):
    """Excess kurtosis, the kurtosis in excess of the Gaussian value of three."""
    return stats.kurtosis(returns)


# Bulk distributional metrics
def empirical_mean(returns):
    """Empirical mean."""
    return np.mean(returns)

def wasserstein2(returns, reference, n_quantiles=1000):
    """Wasserstein-2 distance to the reference sample, computed on a quantile grid."""
    q = np.linspace(0.0005, 0.9995, n_quantiles)
    a = np.quantile(returns, q)
    b = np.quantile(reference, q)
    return np.sqrt(np.mean((a - b)**2))


# Stylised facts
def autocorrelation(series, lag):
    """Autocorrelation of a series at a specified lag."""
    return np.corrcoef(series[:-lag], series[lag:])[0, 1]

def acf_returns(windows, lags=(1, 5, 20)):
    """Mean autocorrelation of raw returns across windows, per lag."""
    return {lag: np.mean([autocorrelation(w, lag) for w in windows]) for lag in lags}

def acf_squared(windows, lags=(1, 5, 20)):
    """Mean autocorrelation of squared returns across windows, per lag."""
    return {lag: np.mean([autocorrelation(w**2, lag) for w in windows]) for lag in lags}

def acf_abs(windows, lags=(1, 5, 20)):
    """Mean autocorrelation of absolute returns across windows, per lag."""
    return {lag: np.mean([autocorrelation(np.abs(w), lag) for w in windows]) for lag in lags}


def all_metrics(windows, process, reference):
    """Compute every evaluation metric for one generation of generated windows.

    The tail and bulk metrics are computed on the pooled returns, while the
    autocorrelations are computed per window and averaged. Errors are absolute
    deviations from the ground truth of the process or the reference sample.

    windows is the generated output with shape n_windows by window_length,
    process provides the ground-truth risk metrics, and reference is a large
    sample from the true process used for the Wasserstein distance.
    Returns a flat dictionary with one value per metric.
    """

    flat = windows.ravel()
    metrics = {
        "var_error": abs(empirical_var(flat) - process.true_var()),
        "es_error": abs(empirical_es(flat) - process.true_es()),
        "kurtosis": kurtosis(flat),
        "mean": empirical_mean(flat),
        "w2": wasserstein2(flat, reference),
        "mean_error": abs(flat.mean() - reference.mean()),
        "var_of_returns_error": abs(flat.var() - reference.var()),
    }
    for lag, value in acf_returns(windows).items():
        metrics[f"acf_returns_lag{lag}"] = value
    for lag, value in acf_abs(windows).items():
        metrics[f"acf_abs_lag{lag}"] = value
    for lag, value in acf_squared(windows).items():
        metrics[f"acf_sq_lag{lag}"] = value
    
    return metrics
    
