"""Stylised-fact distances between historical and synthetic multivariate log returns.

Every function here takes arrays shaped as produced by ``src.utils.inverse_scaling``:

* ``data``        -- ``(n_samples, n_vars, n_steps)``
* ``real_list`` / ``fake_list`` -- a list of ``n_vars`` arrays of shape ``(n_samples, n_steps)``,
  i.e. ``[data[:, i, :] for i in range(n_vars)]``

The functions were moved out of ``example_notebook.ipynb`` unchanged so the notebook
only keeps the plotting and the printing of results.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, kurtosis, ks_2samp, skew, wasserstein_distance
from statsmodels.tsa.stattools import acf, ccf

__all__ = [
    "rolling_window",
    "compute_avg_emd",
    "calculate_distribution_scores",
    "calculate_skew_kurt_diff",
    "calculate_acf_score_mean_vs_mean_no_std",
    "correlation_mean",
    "correlation_loss",
    "ccf_single_sample",
    "ccf_mean",
    "ccf_loss_mean_vs_mean",
    "leverage_effect_loss_mean_vs_mean",
]


def rolling_window(data, window_size):
    """Stack every ``window_size`` long contiguous slice of ``data`` (time first)."""
    n_windows = data.shape[0] - window_size + 1
    windows = np.zeros((n_windows, window_size, data.shape[1]))
    for idx in range(n_windows):
        windows[idx] = data[idx:idx + window_size]
    return windows


def compute_avg_emd(real_data, fake_data, window):
    """Sum over variables of the EMD between the ``window``-day cumulative return
    distributions of ``real_data`` and ``fake_data``.

    Used to pick the generator checkpoint (the epoch with the smallest value).
    """
    emd = 0
    for i in range(real_data.shape[1]):  # Iterate over features
        real_dist = rolling_window(real_data[:, i, :].T, window).sum(axis=1).ravel()
        fake_dist = rolling_window(fake_data[:, i, :].T, window).sum(axis=1).ravel()
        emd += wasserstein_distance(real_dist, fake_dist)
    return emd


def calculate_distribution_scores(real, fake, num_G, windows):
    """EMD, KL, JS and KS between cumulative return distributions, per variable and window.

    Returns a dict of DataFrames indexed by window size; the EMD frame also carries
    the average over variables in column ``EMD_avg``.
    """
    scores = {
        'EMD': np.zeros((num_G, len(windows))),
        'KL': np.zeros((num_G, len(windows))),
        'JS': np.zeros((num_G, len(windows))),
        'KS': np.zeros((num_G, len(windows)))
    }

    for i in range(num_G):
        for j in range(len(windows)):
            real_dist = rolling_window(real[i].T, windows[j]).sum(axis=1).ravel()
            fake_dist = rolling_window(fake[i].T, windows[j]).sum(axis=1).ravel()

            np.random.shuffle(real_dist)
            np.random.shuffle(fake_dist)

            # Calculate EMD
            scores['EMD'][i, j] = wasserstein_distance(real_dist, fake_dist)

            # Calculate KS Statistic
            scores['KS'][i, j], _ = ks_2samp(real_dist, fake_dist)

            # Create histograms to estimate the probability distributions
            real_hist, bin_edges = np.histogram(real_dist, bins=100, density=True)
            fake_hist, _ = np.histogram(fake_dist, bins=bin_edges, density=True)

            # Normalize the histograms to get probability distributions
            real_prob = real_hist / np.sum(real_hist)
            fake_prob = fake_hist / np.sum(fake_hist)

            # Calculate KL Divergence
            kl_divergence = entropy(real_prob + 1e-10, fake_prob + 1e-10)
            scores['KL'][i, j] = kl_divergence

            # Calculate JS Divergence
            js_divergence = jensenshannon(real_prob + 1e-10, fake_prob + 1e-10)
            scores['JS'][i, j] = js_divergence

    df_scores = {}
    for metric, data in scores.items():
        data = np.round(data, decimals=4)
        df_scores[metric] = pd.DataFrame(data.T, index=windows, columns=[f'{metric} {i}' for i in range(num_G)])

    emd_avg = np.mean(scores['EMD'], axis=0)
    df_scores['EMD']['EMD_avg'] = np.round(emd_avg, decimals=4)

    return df_scores


def calculate_skew_kurt_diff(fake_list, real_list):
    """Absolute difference of the sample-averaged skewness and kurtosis, per variable."""
    results = []
    skewness_diffs = []
    kurtosis_diffs = []

    for asset_idx in range(len(real_list)):
        fake_data = fake_list[asset_idx]
        real_data = real_list[asset_idx]

        # Average the skewness and the kurtosis over the samples
        fake_skewness = np.mean([skew(sample) for sample in fake_data])
        fake_kurtosis = np.mean([kurtosis(sample, fisher=False) for sample in fake_data])
        print(fake_kurtosis)
        real_skewness = np.mean([skew(sample) for sample in real_data])
        real_kurtosis = np.mean([kurtosis(sample, fisher=False) for sample in real_data])
        print(real_kurtosis)

        # Difference of the skewness and of the kurtosis
        skewness_diff = np.abs(real_skewness - fake_skewness)
        kurtosis_diff = np.abs(real_kurtosis - fake_kurtosis)

        # Round
        skewness_diff = round(skewness_diff, 4)
        kurtosis_diff = round(kurtosis_diff, 4)

        results.append((skewness_diff, kurtosis_diff))
        skewness_diffs.append(skewness_diff)
        kurtosis_diffs.append(kurtosis_diff)

        print(f'Asset {asset_idx + 1} Skewness Difference: {skewness_diff}')
        print(f'Asset {asset_idx + 1} Kurtosis Difference: {kurtosis_diff}')

    # Average over the variables
    avg_skewness_diff = round(np.mean(skewness_diffs), 4)
    avg_kurtosis_diff = round(np.mean(kurtosis_diffs), 4)

    print(f"\n[mean] Skewness Difference: {avg_skewness_diff}")
    print(f"[mean] Kurtosis Difference: {avg_kurtosis_diff}")

    return results


def calculate_acf_score_mean_vs_mean_no_std(
    real_list,
    fake_list,
    lags=30,
    loss_type='mse'
):
    """Distance between the sample-mean ACF of the real and of the synthetic paths.

    The ACF is computed on the identity, the absolute value and the square of the
    log returns, for lags ``0 .. lags``. The standard deviation across samples is
    not used. Returns the per-variable scores and their mean over variables.
    """
    # The transforms of the log returns
    data_transforms = [lambda x: x, np.abs, np.square]
    titles = ['Identity log returns', 'Absolute log returns', 'Squared log returns']

    acf_scores = {}  # per variable
    # to average each transform's score over the variables ("mean of means")
    avg_scores_per_transform = {title: [] for title in titles}

    n_vars = len(real_list)
    for group_idx in range(n_vars):
        group_scores = {}  # one scalar per transform for this variable

        for transform, title in zip(data_transforms, titles):
            # (1) transform the real and the synthetic paths
            transformed_real = transform(real_list[group_idx])
            transformed_fake = transform(fake_list[group_idx])

            # (2) ACF of every sample
            acf_real = np.array([acf(ts, nlags=lags) for ts in transformed_real])
            acf_fake = np.array([acf(ts, nlags=lags) for ts in transformed_fake])

            # (3) mean ACF of each side
            mean_acf_real = np.mean(acf_real, axis=0)  # shape=(lags+1,)
            mean_acf_fake = np.mean(acf_fake, axis=0)  # shape=(lags+1,)

            # (4) their difference
            diff = mean_acf_fake - mean_acf_real

            # (5) the loss
            if loss_type == 'mse':
                val = np.mean(diff**2)
            elif loss_type == 'mae':
                val = np.mean(np.abs(diff))
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            # (6) a scalar per transform
            group_scores[title] = round(val, 4)

            # (7) kept for the mean over variables
            avg_scores_per_transform[title].append(val)

        acf_scores[f"Group {group_idx+1}"] = group_scores

    # the mean of the means over the variables
    overall_average_scores = {}
    for title in titles:
        mean_of_means = np.mean(avg_scores_per_transform[title])
        overall_average_scores[title] = round(mean_of_means, 4)

    return acf_scores, overall_average_scores


def correlation_mean(data):
    """Sample-mean correlation matrix across variables.

    Parameters:
        data (np.ndarray): shape = (num_samples, num_time_steps, num_features)

    Returns:
        mean_correlation_matrix (np.ndarray): shape = (num_features, num_features)
    """
    num_samples = data.shape[0]
    correlations = []

    for i in range(num_samples):
        sample = data[i]  # shape = (num_time_steps, num_features)
        corr_mat = np.corrcoef(sample, rowvar=False)
        correlations.append(corr_mat)

    mean_correlation_matrix = np.mean(correlations, axis=0)
    return mean_correlation_matrix


def correlation_loss(real_mean_corr, fake_mean_corr, loss_type='mse'):
    """Distance between the real and the synthetic mean correlation matrix.

    ``loss_type`` is one of ``mse``, ``mae``, ``frobenius``; the value is rounded to 4 decimals.
    """
    diff = fake_mean_corr - real_mean_corr

    if loss_type == 'mse':
        loss_val = np.mean(diff**2)
    elif loss_type == 'mae':
        loss_val = np.mean(np.abs(diff))
    elif loss_type == 'frobenius':
        loss_val = np.linalg.norm(diff, ord='fro')
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    return round(loss_val, 4)


def ccf_single_sample(x, max_lag=10):
    """Cross-correlation of every pair of variables of one sample, for lags ``0 .. max_lag``.

    Parameters
    ----------
    x: np.ndarray
      shape = (time_steps, features)

    Returns
    -------
    cc_matrix: np.ndarray
      shape = (num_features, num_features, max_lag+1)
    """
    time_steps, num_features = x.shape
    cc_matrix = np.zeros((num_features, num_features, max_lag+1), dtype=float)

    for i in range(num_features):
        for j in range(num_features):
            # statsmodels ccf: the correlations at lag >= 0
            c_array = ccf(x[:, i], x[:, j], adjusted=False)
            # keep lag=0 .. max_lag
            c_array = c_array[:(max_lag+1)]
            if len(c_array) < (max_lag+1):
                # pad with zeros if shorter
                c_array = np.pad(c_array, (0, max_lag+1 - len(c_array)), 'constant')
            cc_matrix[i, j, :] = c_array

    return cc_matrix


def ccf_mean(data, max_lag=10):
    """Sample-mean of :func:`ccf_single_sample`.

    Parameters
    ----------
    data: np.ndarray
      shape = (num_samples, time_steps, features)

    Returns
    -------
    mean_ccf: np.ndarray
      shape = (num_features, num_features, max_lag+1)
    """
    num_samples, time_steps, num_features = data.shape

    all_cc = np.zeros((num_samples, num_features, num_features, max_lag+1), dtype=float)

    for s_idx in range(num_samples):
        cc_matrix = ccf_single_sample(data[s_idx], max_lag=max_lag)
        all_cc[s_idx] = cc_matrix  # shape=(features, features, max_lag+1)

    mean_ccf = all_cc.mean(axis=0)  # shape=(features, features, max_lag+1)
    return mean_ccf


def ccf_loss_mean_vs_mean(real_mean_ccf, fake_mean_ccf, loss_type='mse'):
    """Distance between the real and the synthetic mean cross-correlation tensor.

    ``loss_type`` is one of ``mse``, ``mae``, ``frobenius``; the value is rounded to 4 decimals.
    """
    diff = fake_mean_ccf - real_mean_ccf

    if loss_type == 'mse':
        loss_val = np.mean(diff**2)
    elif loss_type == 'mae':
        loss_val = np.mean(np.abs(diff))
    elif loss_type == 'frobenius':
        # Frobenius norm of the 3-d array
        loss_val = np.linalg.norm(diff)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    return round(loss_val, 4)


def leverage_effect_loss_mean_vs_mean(real_list, fake_list, lags=30, loss_type='mse'):
    """Distance between the mean leverage effect ``Corr(r_t, r_{t+tau}^2)`` of the two sides.

    Parameters:
        real_list (list of np.ndarray): per variable, shape=(n_real_samples, time_length)
        fake_list (list of np.ndarray): per variable, shape=(n_fake_samples, time_length)
        lags (int): the largest lag tau
        loss_type (str): 'mse', 'mae' or 'frobenius'

    Returns:
        leverage_scores (dict): the per-variable loss
        overall_mean (float): their mean
    """

    def leverage_effect(ts, tau):
        """Corr(r_t, r_(t+tau)^2)"""
        rt = ts[:-tau]            # drop the last tau
        rt_squared = ts[tau:]**2  # drop the first tau, then square
        return np.corrcoef(rt, rt_squared)[0, 1]

    n_vars = len(real_list)
    leverage_scores = {}     # per variable
    all_group_losses = []    # to average over the variables

    for i in range(n_vars):
        real_data = real_list[i]  # shape=(n_real_samples, time_length)
        fake_data = fake_list[i]  # shape=(n_fake_samples, time_length)

        # (1) leverage effect of every real sample, tau = 1 .. lags
        real_leverage_effects = np.array([
            [leverage_effect(ts, tau) for tau in range(1, lags+1)]
            for ts in real_data
        ])  # shape = (n_real_samples, lags)

        # (2) the same for the synthetic samples
        fake_leverage_effects = np.array([
            [leverage_effect(ts, tau) for tau in range(1, lags+1)]
            for ts in fake_data
        ])  # shape = (n_fake_samples, lags)

        # (3) the mean over the real samples
        mean_real_leverage = np.mean(real_leverage_effects, axis=0)  # shape=(lags,)

        # (4) the mean over the synthetic samples
        mean_fake_leverage = np.mean(fake_leverage_effects, axis=0)  # shape=(lags,)

        # (5) their difference
        diff = mean_fake_leverage - mean_real_leverage

        # (6) the loss
        if loss_type == 'mse':
            loss_val = np.mean(diff**2)
        elif loss_type == 'mae':
            loss_val = np.mean(np.abs(diff))
        elif loss_type == 'frobenius':
            loss_val = np.linalg.norm(diff)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        # (7) store, rounded only at the end
        leverage_scores[f'Group {i+1}'] = loss_val
        all_group_losses.append(loss_val)

        # (8) print the per-variable loss
        print(f"[Group {i+1}] Leverage Loss ({loss_type.upper()}) => {loss_val:.4f}")

    # (9) the mean over the variables
    overall_mean = np.mean(all_group_losses)
    print(f"\n[Overall] Leverage Loss ({loss_type.upper()}) => {overall_mean:.4f}")

    # (10) round
    for group in leverage_scores:
        leverage_scores[group] = round(leverage_scores[group], 4)
    overall_mean = round(overall_mean, 4)

    return leverage_scores, overall_mean
