# Seasonal model of Sharma et al. COVID-19 Nonpharmaceutical Interventions Effectiveness

This repository contains one part of the code used in the paper [Gavenčiak et al.: Seasonal variation in SARS-CoV-2 transmission in temperate climates: A Bayesian modelling study in 143 European regions](https://doi.org/10.1371/journal.pcbi.1010435), PLOS Comp. Bio., 2022. The 2021 preprint can be found [here](https://www.medrxiv.org/content/10.1101/2021.06.10.21258647v1).

This repository contains the seasonal variant of the model of [Sharma et al. (2021), *Understanding the effectiveness of government interventions in Europe’s second wave of COVID-19*](https://www.medrxiv.org/content/10.1101/2021.03.25.21254330v1) and has been forked from [MrinankSharma/COVID19NPISecondWave](https://github.com/MrinankSharma/COVID19NPISecondWave); please see that repo for further details.

For the seasonal variant of the model of [Brauner et al. *Inferring the effectiveness of government interventions against COVID-19*](https://www.medrxiv.org/content/10.1101/2020.05.28.20116129v2.article-info), see the repository [gavento/covid_seasonal_Brauner](https://github.com/gavento/covid_seasonal_Brauner).

## Data

The main data file used in the model is `data/modelSharma_dataSharma.csv` which is identical to `data/all_merged_data_2021-01-22.csv` from Sharma et al. except for number formatting and carrying several more (unused) features. 

The data files `data/modelSharma_dataSharma_countryMobility_*.csv` are enriched with [Google community mobility reports](https://www.google.com/covid19/mobility/). The column `Mobility decrease` is a mean of indicated mobility categories remapped to range from 0.0 (no mobility) to 1.0 (pre-pandemic mobility), as described in the paper.

## Running the model

Instructions for recent linux distributions (E.g. Ubuntu 20.04+)

* Install poetry in case you don't already have it (follow instructions at https://python-poetry.org for non-default install or config).

```sh
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python - --version 1.1.6
source $HOME/.poetry/env
```

* Install dependencies into a poetry virtualenv (once)

```sh
poetry install
```

* Run all or selected the inferences

Adjust the number of parralel runs: each paralllel run uses 4 CPU cores.

```sh
# Basic seasonality and sensitivity analysis
poetry run python scripts/sensitivity_dispatcher.py --max_parallel_runs 4 --model_config modelSharma_dataSharma \
  --num_samples 1250 --num_chains 4 --num_warmup 250 \
  --categories seasonality_basic_R_normal_Sharma seasonality_maxRday_normal_Sharma \
  seasonality_maxRday_fixed_Sharma seasonality_local_Sharma basic_R_normal_Sharma default_Sharma

# Mobility sensitivity analysis
poetry run python scripts/sensitivity_dispatcher.py --max_parallel_runs 4 --model_config modelSharma_dataSharma_countryMobility1 \
  --num_samples 1250 --num_chains 4 --num_warmup 250 \
  --categories seasonality_basic_R_normal_Sharma seasonality_maxRday_normal_Sharma
```

* Use notebooks in `notebooks/final_results` to create the plots.

## Changelog

* Preprint v1 (tag [`preprint-v1`](https://github.com/gavento/covid_seasonal_Sharma/releases/tag/preprint-v1))
  * Add seasonality model (also added to upstream)
  * Customized and extended plotters 
  * Runners and configs for sensitivity analyses
  * Fixes and updates

* Preprint v2 (tag [`preprint-v2`](https://github.com/gavento/covid_seasonal_Sharma/releases/tag/preprint-v2))
  * Added mobility sensitivity analysis, data and plotters
  * Rebased to updates in `MrinankSharma/COVID19NPISecondWave` commit [8884dc8](https://github.com/MrinankSharma/COVID19NPISecondWave/commit/8884dc8f9add0b7be6a3c1ee71944c632679f1e0).
    * Now includes real-world data from `MrinankSharma/COVID19NPISecondWave`
  * Update docs and configs for easier reproduction

* Plos Comp. Bio v1 (tag [submitted-1](https://github.com/gavento/covid_seasonal_Sharma/tree/submitted-1))
  * Add sensitivity analyses for the seasonal forcing model and seasoanlity interactions
  * Add plotters for the new analyses and the final plots
  * Update major dependencies (numpyro, JAX, arviz), update code to match

## Questions?

Please email Tomáš Gavenčiak (`gavento` at `ucw` dot `cz`) or Mrinank Sharma (`mrinank` at `robots` dot `ac` dot `uk`, only regarding their code) for questions regarding the codebase.
