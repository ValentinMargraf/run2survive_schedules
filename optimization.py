import random
import sys

import numpy as np
import pandas as pd
from typing import Dict, List

from OppOpPopInit import OppositionOperators
from geneticalgorithm2 import geneticalgorithm2 as ga
from schedule import compute_mean_runtime, survival_curve_from_schedule, schedule_termination_curve


global EVENT_TIMES
global SURVIVAL_FUNCTIONS
global CUTOFF
global PAR_K
global SCENARIO
global BEST_INDEX
global FILTERED_ALGORITHMS

def solution_to_schedule(solution: List[float], packed: bool = True):
    '''
    Returns an algorithm schedule of our structure given a dictionary that contains the different timesteps.
    If the list contains a 0 then the algorithms will not be considered for the schedule.
    '''
    if packed:
        solution_dict: Dict[int, float] = pd.DataFrame(solution['variable'])[0].to_dict()
    else:
        solution_dict: Dict[int, float] = pd.DataFrame(solution)[0].to_dict()
    schedule = list(sorted((elem for elem in solution_dict.items() if elem[1] != 0), key=lambda item: item[1]))

    act_schedule = list()
    for elem in schedule:
        act_schedule.append((FILTERED_ALGORITHMS[elem[0]], elem[1]))

    act_schedule[-1] = (act_schedule[-1][0], -1)  # Change last timestep to -1, so that it fits to our schedule structure
    return act_schedule


def solution_to_runtime_list(X) -> List[float]:
    runtimes = []
    activation = []
    for i in range(len(X)):
        if i % 2 == 0:
            runtimes.append(X[i])
        else:
            activation.append(X[i])

    for i in range(len(activation)):
        if activation[i] == 0:
            runtimes[i] = 0

    return np.array(runtimes)


def fitness_function(X) -> float:
    '''
    Returns a fitness (float) based on the array that contains the current values of every individual.
    This is required for evaluating the individual.
    '''
    try:
        schedule = solution_to_schedule(solution_to_runtime_list(X), False)
    except:
        return sys.float_info.max
    event_times, function_values = survival_curve_from_schedule(schedule, EVENT_TIMES, SURVIVAL_FUNCTIONS, CUTOFF)
    return compute_mean_runtime(event_times, function_values, CUTOFF, PAR_K)


def fitness_function_termination(X) -> float:
    '''
    Returns a fitness (float) based on the array that contains the current values of every individual.
    This is required for evaluating the individual.
    '''
    try:
        schedule = solution_to_schedule(solution_to_runtime_list(X), False)
    except:
        return sys.float_info.max
    event_times, function_values = schedule_termination_curve(schedule, CUTOFF, SCENARIO)
    return compute_mean_runtime(event_times[0], function_values[0], CUTOFF, PAR_K)


def run_optimization(num_algs: int, cutoff: float, max_num_iteration: int = 150, survival: bool = True):
    '''
    This functions creates a genetical optimizer that optimizes a schedule using a fitnessfunction
    '''
    population_size = 100
    algorithm_param = {'max_num_iteration': max_num_iteration,
                       'population_size': population_size,
                       'mutation_probability': 0.1,
                       'elit_ratio': 0.3,
                       'crossover_probability': 0.9,
                       'parents_portion': 0.5,
                       'crossover_type': 'one_point',
                       'max_iteration_without_improv': 100}
    varbound= [[0, cutoff], [0, 1]]*num_algs
    vartype = ['real', 'int']*num_algs

    # create start generation with
    samples = []
    for i in range(num_algs):
        sample = [0.0] * (num_algs * 2)
        alg_index = i * 2
        sample[alg_index] = random.uniform(0, cutoff)
        sample[alg_index+1] = 1.0
        samples.append(np.array(sample))

    while len(samples) < population_size:
        sample = [random.uniform(0, cutoff), random.randint(0,1)] * num_algs
        samples.append(np.array(sample))

    model = ga(function=fitness_function if survival else fitness_function_termination,
               dimension=len(vartype),
               variable_type_mixed=vartype,
               variable_boundaries=varbound,
               algorithm_parameters=algorithm_param)
    model.run(disable_progress_bar=True, disable_printing=True, no_plot=True, start_generation=np.array(samples))# revolution_after_stagnation_step=50, revolution_part= 0.2,
              # revolution_oppositor=OppositionOperators.Continual.quasi(minimums=np.array(varbound)[:, 0], maximums=np.array(varbound)[:, 1]))
    return model.output_dict
