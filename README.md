# Seasonal model of Sharma et al. COVID-19 Nonpharmaceutical Interventions Effectiveness

This repository contains one part of the code used in the paper [Gavenčiak et al.: *Seasonal variation in SARS-CoV-2 transmission in temperate climates*](https://www.medrxiv.org/content/10.1101/2021.06.10.21258647v1). **Note**: this paper is a preprint and has not yet been peer-reviewed.  

This repository contains the seasonal variant of the model of [Sharma et al. (2021), *Understanding the effectiveness of government interventions in Europe’s second wave of COVID-19*](https://www.medrxiv.org/content/10.1101/2021.03.25.21254330v1) and has been forked from [MrinankSharma/COVID19NPISecondWave](https://github.com/MrinankSharma/COVID19NPISecondWave); please see that repo for further details.

For the seasonal variant of the model of [Brauner et al. *Inferring the effectiveness of government interventions against COVID-19*](https://www.medrxiv.org/content/10.1101/2020.05.28.20116129v2.article-info), see the repository [gavento/covid_seasonal_Brauner](https://github.com/gavento/covid_seasonal_Brauner).

# Changelog

* Preprint v1 (tag [`preprint-v1`](https://github.com/gavento/covid_seasonal_Sharma/releases/tag/preprint-v1))
  * Add seasonality model (also added to upstream)
  * Customized and extended plotters 
  * Runners and configs for sensitivity analyses
  * Fixes and updates

* Preprint v2
  * Added mobility sensitivity analysis, data and plotters
  * Rebased to updates in `MrinankSharma/COVID19NPISecondWave` commit [8884dc8](https://github.com/MrinankSharma/COVID19NPISecondWave/commit/8884dc8f9add0b7be6a3c1ee71944c632679f1e0).
  * Added real data from `MrinankSharma/COVID19NPISecondWave`

# Questions?

Please email Tomáš Gavenčiak (`gavento` at `ucw` dot `cz`) or Mrinank Sharma (`mrinank` at `robots` dot `ac` dot `uk`, only regarding their code) for questions regarding the codebase.
