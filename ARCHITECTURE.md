# Architecture

A map of this repository: what each module is for, how the modules depend on each other,
the conventions they follow, and what changing one of them affects.

## Structure

```
MVFIT-GAN/
├── configs/config.yaml         # the only training configuration
├── data/
│   ├── indices.csv             # the training data (6 daily series, 2001-01-02 .. 2024-11-29)
│   └── battery_cycles/         # 4 battery charge/discharge logs, read by no code here
├── example_notebook.ipynb      # the entry point: preprocess, train, select, evaluate
├── assets/                     # README / documentation images
├── requirements.txt
└── src/
    ├── utils.py                # scaling / inverse_scaling, seeds, tensor helpers
    ├── stylized_facts.py       # stylised-fact distances (moved out of the notebook)
    ├── preprocess/
    │   └── gaussianize.py      # Lambert W / Box-Cox Gaussianization (Greg Ver Steeg)
    ├── models/
    │   ├── tcn.py              # TemporalBlock, Chomp1d
    │   ├── generators.py       # UserGenerator  (dilated TCN, noise -> n_vars series)
    │   ├── discriminators.py   # UserDiscriminator (dilated TCN -> one probability)
    │   └── trainer.py          # GANTrainer: the training loop and the checkpoints
    ├── eval_trading/           # evaluation through trading strategies and risk measures
    │   ├── config.yaml         # which metrics are enabled, and their parameters
    │   ├── strategies.py       # log_return_to_price + 4 strategies, PnL trajectories
    │   ├── summary.py          # full_evaluation / EvaluationComponent (the driver)
    │   ├── loss.py             # VaR, ES, max drawdown, cumulative PnL, ACF, histogram, ...
    │   ├── metrics.py          # the same quantities as standalone metric objects
    │   ├── eval_helper.py      # the train/validate/test loop of the learned scores
    │   ├── scores.py           # discriminative score, predictive score
    │   └── metadata            # leftover CodaLab scoring descriptor from upstream
    └── eval_metrics/           # evaluation through distributional metrics and plots
        ├── plot.py             # plot_samples, tsne_plot, compare_acf_matrix, ...
        ├── evaluations.py      # full_evaluation over a dataloader, trains the learned scores
        ├── loss.py             # SigW1Loss, HistoLoss, ACFLoss, CrossCorrelLoss, CovLoss, ...
        ├── metrics.py          # metric objects, incl. signature metrics
        ├── test_metrics.py     # Sig-MMD, predictive FID / KID
        ├── augmentations.py    # path augmentations for the signature (AddTime, LeadLag, ...)
        ├── eval_helper.py      # ACF / cross-correlation helpers, train loops
        ├── scores.py, summary.py
        └── hypothesis_test.py  # NOT importable: needs src/datasets/, absent from this repo
```

`src/eval_trading/` and `src/eval_metrics/` were named `src/evaluation/` and `src/evaluations/`.
They are two different evaluation suites adopted from two different upstream codebases, and
several modules share file names (`loss.py`, `metrics.py`, `eval_helper.py`, `scores.py`,
`summary.py`) with **different** contents, so the old names were a trap. They were renamed, not
merged: neither is a superset of the other.

* `eval_trading` adds `strategies.py`, `MaxDrawbackLoss` and `CumulativePnLLoss`, and its
  `EvaluationComponent` runs every metric on the PnL trajectory of a trading strategy.
* `eval_metrics` adds the signature-based metrics (`SigW1Loss`, `Sig_mmd`), the predictive
  FID / KID, the plots, and the path augmentations, and evaluates the paths directly.

`src/baselines/` was renamed `src/models/` (it holds the model of this repository, not
baselines) and its `networks/` level was flattened.

## Dependencies

Internal import graph (`A -> B` means A imports B):

```
example_notebook.ipynb
  ├─> src.utils ──────────────> src.preprocess.gaussianize
  ├─> src.stylized_facts
  ├─> src.models.generators ──┐
  ├─> src.models.discriminators ├─> src.models.tcn
  ├─> src.models.trainer ─────┘
  ├─> src.eval_trading.summary ─> src.eval_trading.{loss -> metrics -> eval_helper,
  │                                                 scores -> eval_helper, strategies}
  ├─> src.eval_trading.strategies
  ├─> src.eval_metrics.plot ────> src.eval_metrics.eval_helper
  └─> src.eval_metrics.evaluations ─> src.eval_metrics.loss ─> src.eval_metrics.{eval_helper,
                                                                metrics, test_metrics}
```

Fan-in (how many modules import it) — the de facto interfaces:

| Module | Imported by | Why it matters |
|---|---|---|
| `src/utils.py` | 9 modules + the notebook | `scaling` / `inverse_scaling` define the data contract of the whole pipeline; `to_numpy`, `AddTime`, `set_seed`, `loader_to_tensor` are used everywhere |
| `src/models/tcn.py` | generator and discriminator | one `TemporalBlock` shapes both networks |
| `src/eval_metrics/eval_helper.py` | 4 modules of its package | ACF / cross-correlation primitives and the train loop of the learned scores |
| `src/eval_trading/eval_helper.py` | 2 modules of its package | the same role in the other suite |
| `src/preprocess/gaussianize.py` | `src/utils.py` | the Gaussianization step of `scaling` |

External dependencies: `torch`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`,
`matplotlib`, `seaborn`, `tqdm`, `pyyaml`, `ml_collections`; `signatory` for the signature
metrics of `src/eval_metrics`; `wandb` is imported by `src/eval_metrics/evaluations.py` and
at the top of the notebook but nothing logs to it. See `requirements.txt`.

## Conventions

* **Tensor layouts.** The notebook keeps `(n_samples, n_steps, n_vars)` while the networks and
  `inverse_scaling` work on `(n_samples, n_vars, n_steps)`; the notebook transposes at each
  boundary (`training_data.transpose(1, 2)`). `src/stylized_facts.py` takes the
  `(n_samples, n_vars, n_steps)` form, or a list of `n_vars` arrays `(n_samples, n_steps)`.
* **Configuration.** `configs/config.yaml` is read into an `ml_collections.ConfigDict` in the
  notebook and passed whole to the generator, the discriminator and the trainer; each reads
  the attributes it needs. `src/eval_trading/config.yaml` is a second, separate config read
  only by `full_evaluation`.
* **Checkpoints.** `GANTrainer` derives its own output directory from the configuration:
  `results/models/{file_name}_{n_epochs}_{batch_size}_G_lr_{lr_G}_D_lr_{lr_D}_noise_{noise_dim}_hidden_dim_{G_hidden_dim}/Generator_{epoch}.pt`,
  written every 5 epochs. The notebook rebuilds the same string by hand (`full_name`) to load
  them back, so renaming a configuration key breaks loading as well as saving.
* **Naming.** lower_snake_case modules and functions, CapWords classes. The networks are
  called `UserGenerator` / `UserDiscriminator` after the upstream template they replace.
* No tests, no CI, no package metadata; `src` is an implicit namespace package, so everything
  must be run from the repository root for `import src...` to resolve.

## Impact

What to expect when touching each part.

| Change | Reaches |
|---|---|
| `src/utils.py` `scaling` / `inverse_scaling` | every result: training targets, the checkpoint selection, and both evaluation suites |
| `configs/config.yaml` (`n_vars`, `n_steps`, `noise_dim`, hidden dims) | network shapes, the checkpoint directory name, and the shape of everything downstream; old checkpoints stop loading |
| `src/models/tcn.py` | generator and discriminator together; the dilations `[1, 2, 4, 8, 16, 32]` with kernel 2 give a receptive field of 127 steps, matched to `n_steps = 128` |
| `src/eval_trading/strategies.py` | every number of `full_evaluation`, since all its metrics are computed on strategy PnL |
| `src/stylized_facts.py` | the stylised-fact tables of the notebook only |
| renaming a config key | both `GANTrainer.results_dir` and the notebook's `full_name` string |

Known rough edges, left as they are:

* `src/eval_metrics/hypothesis_test.py` imports `src.datasets.AR1` and `src.datasets.GBM`,
  which do not exist in this repository; the module cannot be imported.
* `src/eval_trading/config.yaml` enables only `var`, `es`, `max_drawback` and
  `cumulative_pnl`. The other metrics of `EvaluationComponent` (`discriminative_score`,
  `predictive_score`, `sigw1`, `permutation_test`, ...) read configuration keys that this file
  does not define, so enabling them needs the config extended first.
* `config.yaml` carries three keys nothing reads: `output_dim`, `data_dir`, `result_dir`. The
  data path (`./data/{file_name}.csv`) and the results path are hardcoded instead.
* `src/utils.py` binds `Tensor = torch.cuda.FloatTensor` at import time, and
  `deterministic_NeuralSort` hardcodes `"cuda"`. Neither is used by the notebook.
* `src/eval_trading/metadata` is an upstream CodaLab scoring descriptor, kept for provenance.
* `src/eval_trading/strategies.py` builds the PnL with Python loops over batch and asset
  (`execute_trades_at_t`), which dominates the cost of `full_evaluation`.
* The notebook's train / test split is a random 70/30 partition of **overlapping** rolling
  windows, and the evaluation compares the synthetic paths against the *training* windows
  (`real_data = inverse_scaling(training_data...)`), not against `test_data`. Before that
  comparison it also drops synthetic windows that leave 1.5 × the historical per-series range.
  This is how the stored results were produced and was left unchanged.
* Nothing in the repository expands the name `MVFIT-GAN`; the model is the multivariate
  (6-series) Quant GAN described in the README.

### Complexity ranking

There are **no tests** in this repository, so a true CRAP score cannot be computed; the table
below ranks `CRAP_max = cc² + cc` (the score a function would get at zero coverage), from
`lizard` over `src/` — 345 functions, mean cyclomatic complexity 1.99.

| CRAP_max | CC | NLOC | Function |
|---|---|---|---|
| 210 | 14 | 127 | `src/eval_metrics/evaluations.py :: full_evaluation` |
| 182 | 13 | 45 | `src/eval_metrics/summary.py :: eval_summary` |
| 132 | 11 | 52 | `src/eval_metrics/evaluations.py :: _train_classifier` |
| 110 | 10 | 44 | `src/eval_trading/eval_helper.py :: update_per_epoch` |
| 110 | 10 | 35 | `src/eval_trading/summary.py :: eval_summary` |
| 90 | 9 | 58 | `src/eval_metrics/evaluations.py :: train_predictive_FID_model` |
| 72 | 8 | 42 | `src/eval_metrics/evaluations.py :: _train_regressor` |
| 72 | 8 | 25 | `src/eval_trading/strategies.py :: get_pnl_trajectory` |
| 72 | 8 | 12 | `src/preprocess/gaussianize.py :: transform` |

The risk sits entirely in the two evaluation suites, not in the model: `src/models/` peaks at
CC 6 (`TemporalBlock.__init__`). Of the functions above, only `eval_trading`'s `eval_summary`,
`update_per_epoch` and `get_pnl_trajectory` are on the path the notebook actually runs.
