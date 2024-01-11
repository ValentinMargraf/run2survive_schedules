# Code for paper: "RunAndSchedule2Survive: Algorithm Scheduling Based on Run2Survive"

This repository contains code for the paper "RunAndSchedule2Survive: Algorithm Scheduling Based on
Run2Survive". We leverage Run2Survive runtime distributions to construct algorithm schedules.


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
the de-facto standard benchmark for algorithm selection and scheduling, we demonstrate that RunAndSched-
ule2Survive achieves best performance in 21 out of 25 benchmark scenarios over the hitherto state-of-the-art
approaches.

## Setup

```bash
conda create --name r2s_schedules python=3.8
conda activate r2s_schedules
pip install -r requirements.txt
```

## Running the code
The relevant code of our implementation is contained in the file run_schedule2.py
