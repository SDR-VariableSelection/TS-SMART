
# Sufficient dimension reduction for high-dimensional nonlinear vector autoregressive models

Here we include the Python files that can reproduce all the figures and tables in the main text and appendix.  

## Description

Description of files:

* **simulation.py** : first runs this file to obtain experimental results for Examples 1--4, saved in the file "simulation.pkl". 
* **simulation_plot.ipynb** : after having the "simulation.pkl" from **simulation.py** file, run it to generates the line plots, which produce Figures 1-4. This file also provide results for Tables 1 and 2. 
* **macro_data.ipynb**: conducts real data analysis on macroeconomic data and provides results for Table 2. 

support files
* **lib_gpu.py** : uses cupy and provides all supporting functions for TS-MART, TS-SMART, TS-SIR, and TS-RR in the simulations.
* **_functions.py**: uses nupy and provides functions for real data analysis. 

## Hardware
All the simulations were run on the GPU:NVIDIA A100-PCIE-40GB equipped with a total of 40 GB of RAM. 

## Execution time
* We used Python 3.11.4 and CUDA 12.1 to run all the simulations. 

* The total execution times of script **simulation.py** for 100 repetitions over 16 settings: four examples (1--4) and four sample sizes from 500 to 3000 is about 21 hours. 

* For real data analysis, we used the local computer with Python 3.12.0. The execution time is less than 1 min. 
