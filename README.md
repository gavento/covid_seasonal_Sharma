# Seasonal model of Sharma et al. COVID-19 Nonpharmaceutical Interventions Effectiveness

This repository contains one part of the code used in the paper [Gavenčiak et al.: *Seasonal variation in SARS-CoV-2 transmission in temperate climates*](https://www.medrxiv.org/content/10.1101/2021.06.10.21258647v1). **Note**: this paper is a preprint and has not yet been peer-reviewed.  

This repository contains the seasonal variant of the model of [Sharma et al. (2021), *Understanding the effectiveness of government interventions in Europe’s second wave of COVID-19*](https://www.medrxiv.org/content/10.1101/2021.03.25.21254330v1) and has been forked from [MrinankSharma/COVID19NPISecondWave](https://github.com/MrinankSharma/COVID19NPISecondWave); please see that repo for further details.

For the seasonal variant of the model of [Brauner et al. *Inferring the effectiveness of government interventions against COVID-19*](https://www.medrxiv.org/content/10.1101/2020.05.28.20116129v2.article-info), see the repository [gavento/covid_seasonal_Brauner](https://github.com/gavento/covid_seasonal_Brauner).

# Data

**This repo currently only contains synthetic data. The real data is available upon request. Please email Jan Brauner at jan.brauner at cs dot ox dot ac dot uk**

# Reproducing Results
An example main model run is in ``notebooks/final_results/main_results.ipynb``. 

```
import numpyro
numpyro.set_host_device_count(4)

from epimodel import preprocess_data, run_model, EpidemiologicalParameters, default_model

# load data
data = preprocess_data('../../data/all_merged_data_2021-01-22.csv')
data.featurize() # convert raw NPI data to binary features
data.mask_new_variant(new_variant_fraction_fname='../../data/nuts3_new_variant_fraction.csv') # mask VOC
data.mask_from_date('2021-01-09') #mask end of perior

ep = EpidemiologicalParameters() # load delay dists

samples, warmup_samples, info, mcmc = run_model(default_model, data, ep, num_samples=1250, target_accept=0.75, num_warmup=250, num_chains=4, max_tree_depth=15) # run model

```

For the sensitivity analyses, sensitivity analysis scripts are in `scripts/sensitivity_analysis`. Each script define a particular sensitivity analysis type. Many sensitivity analyses can be launched in parallel using `sensitivity_dispatcher.py`. 

For example, you can run the full set of sensitivity analyses on a server as follows:
```
python scripts/sensitivity_dispatcher.py --max_parallel_runs NRUNs --model_type MODEL --model_config CONFIG --categories [CATEGORY_1 CATEGORY_2 ... ]
# example    
python scripts/sensitivity_dispatcher.py --max_parallel_runs 24 --model_type default --model_config default --categories npi_leaveout output_noise_scale_prior r_walk_noise_scale_prior r_walk_period seeding_days seeding_scaling basic_R_prior_mean basic_R_prior_scale england_ifr_iar cases_delay_mean death_delay_mean gen_int_mean frac_voc infection_noise_scale intervention_prior bootstrap
```
**Note**: the parallelisation currently only works on Unix systems, since it requires the use of `taskset` to correctly parallelise the numpyro code. 

* Sensitivity analysis options for a given script are input using Python `argparse` arguments. e.g., in `seeding_days`, there is an argparse argument `--n_days_seeding N_DAYS` which is the number of seeding days.  
* `model_configs.yaml` contains model configs and featurise configs that can be used to overwrite model arguments across many sensitivity analyses. For example, if you wanted to run a full sensitivity analysis suite with a different model input argument, you could use the config. 
* `sensitivity_analysis.yaml` defines the input arguments used for the sensitivity analysis scripts, and sensitivity analysis categories. The arguments in the yaml file are exactly the argparse arguments of the scripts. 

All plotting code is in the `notebooks/final_results` folder. See `validation_plotter` (Fig. 3), `main_result_plotter` (Fig. 1, other plots also) and `sensitivity_analysis_plotter` (Appendix plots). 

# Questions?

Please email Tomáš Gavenčiak (`gavento` at `ucw` dot `cz`) or Mrinank Sharma (`mrinank` at `robots` dot `ac` dot `uk`) for questions regarding the codebase.
