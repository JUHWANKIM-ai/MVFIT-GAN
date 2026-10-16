# MVFIT-GAN — multivariate Quant GAN for financial time series

[![Python](https://img.shields.io/badge/python-3.10-3776AB?logo=python&logoColor=white)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-TCN%20GAN-EE4C2C?logo=pytorch&logoColor=white)](src/models)
[![Data](https://img.shields.io/badge/data-6%20daily%20indices%202001--2024-1f6feb)](data/indices.csv)

A generative model for **multivariate** daily financial log returns: one temporal convolutional
network generates six correlated series jointly, so that the synthetic paths reproduce not only
the fat tails and the volatility clustering of each series but also the correlation between
them. The architecture and the preprocessing follow **Quant GANs** (Wiese, Knobloch, Korn and
Kretschmer, *Quant GANs: deep generation of financial time series*, Quantitative Finance 2020)
— `configs/config.yaml` still names the algorithm `QuantGAN` — extended from one series to a
panel of six, which is what this repository adds.

The data is `data/indices.csv`: daily closes of **DJI, NASDAQ, JPM, HSI, GOLD and WTI_OIL**,
2001-01-02 to 2024-11-29 (5,721 rows). The returns are Gaussianized with a Lambert W transform
before training and mapped back afterwards, so every evaluation is run on real-scale log
returns. The synthetic paths are then judged in two ways: against the stylised facts of the
historical returns (distribution, skewness and kurtosis, autocorrelation, correlation and
cross-correlation, leverage effect), and by running four trading strategies on the generated
prices and comparing the risk of the resulting PnL (VaR, expected shortfall, maximum drawdown,
cumulative PnL).

Training is driven from `example_notebook.ipynb`; there is no command-line entry point.

## Pipeline

```
data/indices.csv                     6 daily closes, 2001-01-02 .. 2024-11-29 (5,721 rows)
  │
  ▼  log difference
log returns                          (5,720, 6)
  │
  ▼  src/utils.py :: scaling         per series: StandardScaler -> Gaussianize (Lambert W) -> StandardScaler
scaled returns
  │
  ▼  rolling window, n_steps = 128
windows (5,593, 128, 6)              random 70 / 30 split -> 3,915 train | 1,678 test
  │
  ▼  src/models/trainer.py :: GANTrainer      300 epochs, batch 256, Adam (G 2e-4 / D 1e-4),
  │                                           BCE loss, D weights clipped to +-0.01,
  │                                           checkpoint every 5 epochs
  ├── G: src/models/generators.py     noise (6, 128) -> dilated TCN, skip connections -> (6, 128)
  └── D: src/models/discriminators.py (6, 128) -> dilated TCN -> spectral-norm conv -> P(real)
  │
  ▼  src/utils.py :: inverse_scaling   back to real-scale log returns
  ▼  checkpoint selection            the epoch with the smallest EMD between the 100-day
  │                                  cumulative-return distributions of real and synthetic paths
synthetic log returns (n, 6, 128)
  │
  ├─▶ src/stylized_facts.py          EMD / KL / JS / KS per horizon, skewness and kurtosis,
  │                                  ACF, correlation, cross-correlation, leverage effect
  ├─▶ src/eval_trading/              log returns -> prices -> 4 strategies (equal weight,
  │                                  mean reversion, trend following, volatility trading)
  │                                  -> VaR, ES, max drawdown, cumulative PnL (alpha = 0.05)
  └─▶ src/eval_metrics/              sample plots, t-SNE of the price paths, signature and
                                     distributional metrics, discriminative / predictive score
```

## Results

Read from the stored outputs of `example_notebook.ipynb` for the run
`indices_300_256_G_lr_0.0002_D_lr_0.0001_noise_6_hidden_dim_120`, 3,915 historical windows
against the same number of synthetic windows. Lower is better everywhere.

**Checkpoint selection.** Scanning the checkpoints from epoch 50 to 295 in steps of 5, the
summed Earth Mover's Distance between the 100-day cumulative-return distributions of the six
series falls from 13.894 (epoch 50) to **0.265 at epoch 230**, the checkpoint used below.

**Cumulative-return distributions**, averaged over the six series:

| Horizon | EMD (mean of 6) |
|---|---|
| 1 day | 0.0007 |
| 5 days | 0.0082 |
| 20 days | 0.0177 |
| 100 days | 0.0376 |

Per-series KL, JS and KS statistics at the same horizons are in the notebook output; at the
1-day horizon they stay within KL 0.0058–0.0132, JS 0.0381–0.0578, KS 0.0119–0.0288.

**Stylised facts**, over the six series:

| Quantity | Distance | Value |
|---|---|---|
| Skewness | mean abs. difference | 0.3702 |
| Kurtosis | mean abs. difference | 6.8981 |
| ACF, identity returns | MSE of the mean ACF, lags 0–30 | 0.0070 |
| ACF, absolute returns | MSE of the mean ACF, lags 0–30 | 0.0020 |
| ACF, squared returns | MSE of the mean ACF, lags 0–30 | 0.0011 |
| Correlation matrix (6×6) | MAE / Frobenius / MSE | 0.0610 / 0.5195 / 0.0075 |
| Cross-correlation, lags 0–10 | MAE / Frobenius / MSE | 0.0755 / 2.1359 / 0.0115 |

The kurtosis is the weak point: the synthetic series are consistently heavier-tailed than the
historical ones (e.g. 13.21 against 4.62 for the first series), while the volatility clustering
is captured well — the ACF distance on squared returns is an order of magnitude below the one
on the returns themselves.

The leverage-effect run was interrupted after three of the six series (MSE 0.0017, 0.0016,
0.0007), so no overall figure for it is stored, and the trading-strategy evaluation
(`full_evaluation`) carries no stored output at all; neither is reported here.

Two properties of this evaluation are worth knowing when reading the tables. The historical
reference is the **training** split, not the held-out one (`real_data` is built from
`training_data`), and the 70/30 split is drawn at random over overlapping rolling windows, so
the two splits overlap in time anyway. And before the comparison, synthetic windows containing
a value outside 1.5 × the per-series historical range are discarded, which removes part of the
generator's tail.

## Repository layout

| Path | Content |
|---|---|
| `example_notebook.ipynb` | the entry point: preprocessing, training, checkpoint selection, every evaluation and figure |
| `configs/config.yaml` | the training configuration (data file, 6 variables, 128 steps, 300 epochs, learning rates) |
| `data/indices.csv` | the training data: 6 daily closes, 2001-01-02 to 2024-11-29 |
| `data/battery_cycles/` | 4 battery pack charge/discharge cycle logs (24 cells, 35 °C and 45 °C, 0.5C/1.5C discharge). Unrelated to the index data and read by no code in this repository; kept as they were committed |
| `src/utils.py` | `scaling` / `inverse_scaling` (the data contract of the pipeline), seeding, tensor helpers |
| `src/preprocess/gaussianize.py` | Lambert W / Box-Cox Gaussianization (utility code by Greg Ver Steeg) |
| `src/models/tcn.py` | `TemporalBlock`: dilated causal convolution, batch norm, LeakyReLU, residual and skip output |
| `src/models/generators.py` | `UserGenerator`: 7 temporal blocks, dilations 1–32, noise → 6 series |
| `src/models/discriminators.py` | `UserDiscriminator`: the mirror network, spectral-norm head → one probability per window |
| `src/models/trainer.py` | `GANTrainer`: the training loop, weight clipping, periodic checkpoints |
| `src/stylized_facts.py` | the stylised-fact distances used by the notebook (EMD/KL/JS/KS, skew and kurtosis, ACF, correlation, cross-correlation, leverage effect) |
| `src/eval_trading/` | evaluation through trading strategies: `strategies.py`, risk measures, and `full_evaluation` driven by `src/eval_trading/config.yaml` |
| `src/eval_metrics/` | evaluation through distributional and signature metrics, plus the sample and t-SNE plots |
| `assets/preprocess_wiese_et_al.png` | screenshot of the preprocessing described in Wiese et al., kept as the reference for `scaling` / `inverse_scaling` |
| `requirements.txt` | the third-party imports of `src/` and the notebook, unpinned |
| `ARCHITECTURE.md` | module roles, import graph, conventions, complexity ranking |

## Get Started

**1. Install.** The notebook was last executed on CPython 3.10.8 with a CUDA build of PyTorch;
no exact versions are recorded in this repository, so `requirements.txt` is unpinned.

```bash
pip install -r requirements.txt
```

`signatory` is needed only by `src/eval_metrics` and must be built against the PyTorch version
already installed; the rest of the pipeline (training, `src/stylized_facts.py`,
`src/eval_trading`) runs without it.

**2. Data.** `data/indices.csv` is in the repository. To train on another panel, put a CSV with
a `Date` column and one numeric column per series in `data/`, and set `file_name` and `n_vars`
in `configs/config.yaml`.

**3. Run.** Everything happens in the notebook, from the repository root (`import src...`
resolves relative to it):

```bash
jupyter lab example_notebook.ipynb
```

* **Data Preprocess** / **Data Construction**: load `configs/config.yaml`, build the scaled
  rolling windows and split them.
* **Model training**: the `trainer.fit()` call is commented out so the notebook can be reopened
  without retraining. Uncomment it to train; checkpoints are written every 5 epochs to
  `results/models/<config string>/Generator_<epoch>.pt` (`results/` is not version controlled).
* **Model Evaluation**: set `full_name` to the directory `GANTrainer` created for your
  configuration, then the next cell picks the checkpoint by the 100-day cumulative-return EMD.
* The remaining cells generate paths, invert the scaling, and run the evaluations and figures.
  Several cells of **Evaluation Phase 2** are commented out in the stored notebook.
