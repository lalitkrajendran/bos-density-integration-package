This code package implements the density integration methodology for BOS outlined in:

1) Rajendran, L. K., Zhang, J., Bhattacharya, S., Bane, S., & Vlachos, P. (2019). Uncertainty quantification in density estimation from background oriented schlieren (BOS) measurements.Measurement Science and Technology. doi:10.1088/1361-6501/ab60c8

2) Rajendran, L., Zhang, J., Bane, S., & Vlachos, P. (2020). Uncertainty-based weighted least squares density integration for background-oriented schlieren. Experiments in Fluids, 61(11), 239. https://doi.org/10.1007/s00348-020-03071-w

sample_script.py is a sample python script that loads the sample dataset in 'sample-data.mat' and calls the density integration + uncertainty quantification function to perform the calculations.
It saves the result to 'sample-result.mat' and a figure to 'sample-result.png'

Please cite the above paper if you use this code package for your work.

