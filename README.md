# Sampling_experiments_LET_USV

This repository contains code used for the study "Assessing improvements in global ocean pCO2 machine learning reconstructions with Southern Ocean autonomous sampling" (Heimdal et al., 2023, https://doi.org/10.5194/bg-2023-160). In this paper, we reconstruct surface ocean pCO2 using the Large Ensemble Testbed (Gloege et al., 2021, https://doi.org/10.1029/2020GB006788) and the pCO2-Residual method (Bennington et al., 2022, https://doi.org/10.1029/2021MS002960). In addition to a "SOCAT-baseline", we tested 11 different sampling experiments with different sampling patterns in the Southern Ocean by USV Saildrones (SOCAT+USV sampling). 

This repository contains code for running the machine learning models and processing data. The reconstructed pCO2 model fields are not published due to large file sizes (75 reconstructions per sampling experiment), however please feel free to reach out if you are interested in any of these. Model output of the Large Ensemble Testbed can be found here: https://figshare.com/collections/Large_ensemble_pCO2_testbed/4568555. A Zonodo link to access the SOCAT+USV sampling masks will be provided.    

Overview of files in this repository:

Data processing and machine learning:

"Calculate_pco2_components.ipynb": Code for calculating pCO2, pCO2-T (direct effect of temperature on pCO2) and pCO2-Residual. We use pCO2-Residual as a target variable for the reconstruction.

"LET_dataframe_XGB.ipynb": Code for setting up data and performing the machine learning for reconstructing pCO2.  

"Calculate_flux.ipynb": Code for calculating air-sea CO2 fluxes from the reconstructed pCO2 model fields and the LET "model truth" fields.

Figures:

"Fig_Boxplots_bias_RMSE.ipynb":

"Fig_boxplots_full_unseen_train_test.ipynb"

"Fig_hovmoller_maps_RMSE_bias.ipynb"

"Fig_hovmoller_map_summer_winter_RMSE_bias.ipynb"

"Fig_maps_full_train_unseen.ipynb"

"Fig_sampling_masks.ipynb" 

"Fig_timeseries_bias_RMSE.ipynb"

"Fig_timeseries_spread_bias_rmse.ipynb"

Supporting files with functions:

"Val_mapping.ipynb": To create map plots.

"pre_saildrone_5.py": Needed to run the data processing and machine learning notebooks. 

Environment file:

"environment_file.yml": list of packages and versions needed to set up an environment to run the code.

