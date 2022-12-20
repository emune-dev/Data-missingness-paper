# Data-missingness-paper
This repository contains all code and simulation scripts for the paper "Missing data in amortized simulation-based neural posterior estimation". It is divided into folders dedicated to the conducted numerical experiments.

Regarding the content of these folders:

- The Jupyter notebooks were used to validate/compare the performance of trained BayesFlow networks and to create illustrative figures for the paper, including convergence plots, posterior plots, SBC plots, etc.
- Subfolders with the ending "ckpts" contain the Python script for training a BayesFlow workflow on a specific forward model, an output file of the running loss as well as the stored networks after the final training epoch.
- Subfolders with the name "bayesflow" contain the implementation of the BayesFlow method downloaded from https://github.com/stefanradev93/BayesFlow. In some cases, slight modifications have been made to meet the purpose of our numerical experiment.
