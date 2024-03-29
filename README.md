# Code for paper: "RunAndSchedule2Survive: Algorithm Scheduling Based on Run2Survive"

This repository contains code for the paper "RunAndSchedule2Survive: Algorithm Scheduling Based on
Run2Survive". This code leverages Run2Survive runtime distributions to construct algorithm schedules.


## Abstract
Algorithm selection denotes the problem of choosing the most suitable algorithm from an algorithm portfolio
for a certain instance of an algorithmic problem class. Here, suitability is often quantified in terms of a
performance metric such as algorithm runtime, which is commonly considered for problem classes such as
Boolean satisfiability or traveling salesperson problems. As a good portfolio exhibits algorithms behaving
complementary in terms of the performance metric on the space of problem instances, algorithm selection
methods hold the potential for speed ups compared to always employing the algorithm being best on average.
At the same time, since algorithms may exhibit extremely long runtimes, cutoff times are imposed to cancel
the execution of an algorithm early. Resulting in so-called right-censored training data, methods from survival
analysis that can naturally handle such right-censored data have demonstrated state-of-the-art performance
for algorithm selection. As a generalization of algorithm selection, algorithm scheduling goes even a step
further and allows selecting a schedule of algorithms instead of a single algorithm per instance, offering
even more potential for improvement. In this work, we propose a novel algorithm scheduling method, called
RunAndSchedule2Survive, leveraging the inherently probabilistic nature of the state-of-the-art algorithm
selection approach Run2Survive to compute well-performing schedules. In an extensive experimental study on
the de-facto standard benchmark for algorithm selection and scheduling, we demonstrate that RunAndSchedule2Survive achieves best performance in 21 out of 25 benchmark scenarios over the hitherto state-of-the-art
approaches.

## ASLib Data
Download the ASlib data from **https://github.com/coseal/aslib_data/tree/master** and place the scenarios into a directory named aslib_data. (Or just simply execute **git clone https://github.com/coseal/aslib_data.git** and the folder with the files will be downloaded into the root directory.)

## Setup
1. Ensure that CMake is installed on your machine.
2. Create a Conda environment using Python 3.7 and install the required libraries:

```bash
conda create --name r2s_schedules python=3.7
conda activate r2s_schedules
pip install -r requirements.txt
```

## Running the code
The relevant code for our implementation is in the file **run_schedule.py**. Assuming, your Conda environment is activated, execute the following command:

```bash
python run_schedule.py

```
This script runs over all scenarios, folds, train-test splits, and calculates the PAR10 score. The results will be saved as **r2ss_par_10.csv** in the **/results** folder.

## Generate Table 2
In the results **/results** folder, you'll find a CSV file named **sbs_vbs_results.csv**, containing the results of the SBS and VBS. To create Table 2 of the paper, showcasing the averaged and normalized PAR10 scores along with their standard deviations, execute the following commands.

```bash
python normalize_par_10.py
python generate_results_table.py

```
The Python script will leverage the pandas library to convert the dataframe into LaTeX table format using the method to_latex(), and the result will be printed.
