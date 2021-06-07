# Seasonal model of Sharma et al. COVID-19 Nonpharmaceutical Interventions Effectiveness

This repository contains one part of the code used in the paper [Gavenčiak et al.: *Seasonal variation in SARS-CoV-2 transmission in temperate climates*](TODO). **Note**: this paper is a preprint and has not yet been peer-reviewed.  

This repository contains the seasonal variant of the model of [Sharma et al. (2021), *Understanding the effectiveness of government interventions in Europe’s second wave of COVID-19*](https://www.medrxiv.org/content/10.1101/2021.03.25.21254330v1) and has been forked from [MrinankSharma/COVID19NPISecondWave](https://github.com/MrinankSharma/COVID19NPISecondWave); please see that repo for further details.

For the seasonal variant of the model of [Brauner et al. *Inferring the effectiveness of government interventions against COVID-19*](https://www.medrxiv.org/content/10.1101/2020.05.28.20116129v2.article-info), see the repository [gavento/covid_seasonal_Brauner](https://github.com/gavento/covid_seasonal_Brauner).

# Data

**This repo currently only contains synthetic data. The real data is available upon request. Please email Jan Brauner at jan.brauner at cs dot ox dot ac dot uk**

# Major changes

The code has been extended with a seasonal model (`seasonality_model`), result and sensitivity analyses plotting scripts (also used for Brauner et al. model plots), and various minor updates.

The code has also been extended to work with several other datasets (brauner et al., Banholzer et al.), although that is not used in the paper and the datasets are not included.

# Questions?

Please email Tomáš Gavenčiak (`gavento` at `ucw` dot `cz`) or Mrinank Sharma (`mrinank` at `robots` dot `ac` dot `uk`) for questions regarding the codebase.
