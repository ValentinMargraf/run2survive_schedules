import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from copy import deepcopy
from collections import Counter


def ground_truth_evaluation(schedule: List[Tuple[int, float]], ground_truth_values: Dict[int, float], cutoff: float, penalty_value: float = -1) -> float:
    '''
    This method computes the actual runtime of an algorithm schedule given the ground truth values for each algorithm
    on a problem instance. It checks wether an algorithm terminates within its timeslot and if the runtime is lower than
    the given cutoff. If no algorithm could solve the problem instance given the schedule, the runtime return will be -1.
    Furthermore, this function works also if it is possible to pause an algorithm and continue with it in another timeslot.
    '''
    runtime = 0
    for alg_id, end_of_timeslot in schedule:
        alg_runtime = ground_truth_values[alg_id]
        total_runtime = runtime + alg_runtime
        timeslot = end_of_timeslot - runtime

        # Check if algorithm terminates within the given timeslot or until the cutoff
        if alg_runtime <= timeslot or (end_of_timeslot == -1 and total_runtime <= cutoff):
            return total_runtime

        runtime = end_of_timeslot
        ground_truth_values[alg_id] = alg_runtime - \
            timeslot  # For paused algorithms

    return penalty_value


def survival_curve_from_schedule(schedule: List[Tuple[int, float]], event_times: List[List[float]], survival_functions: List[List[float]], cutoff: float):
    '''
    Given the survival curve data from the sole algorithms this method create a new survival curve according
    to a also given schedule. Furthermore the new created schedule also ends at the cutoff.
    If two algorithms have the same endpoint of time than the algorithm with the lower index will be considered.
    This is usually done for a comparing the schedule to the single algorithms on a certain problem instance.
    '''
    event_times, survival_functions = deepcopy(event_times), deepcopy(survival_functions) 
    new_event_times, new_survival_functions = np.array([]), np.array([])
    total_runtime, end_of_timeslot_probability = 0, 1.0
    for alg_id, end_of_timeslot in schedule:
        timeslot = (end_of_timeslot if end_of_timeslot != -
                    1 else cutoff) - total_runtime

        # In case to algorithms are positioned for the same point of time
        if timeslot == 0:
            continue

        # Search for index where t1 <= timeslot < t2
        index_endpoint_timeslot = np.searchsorted(event_times[alg_id], timeslot, 'right')
        index_event_probabilities = -np.searchsorted(np.flip(survival_functions[alg_id]), end_of_timeslot_probability)
        # assert (end_of_timeslot_probability != 1.0) or (survival_functions[alg_id][index_event_probabilities] == survival_functions[alg_id][0]) 

        # Append new events (shifted by the current total runtime) and probabilities to the output survival function
        if index_event_probabilities != 0:
            new_event_times = np.concatenate((new_event_times, np.array(event_times[alg_id][index_event_probabilities:index_endpoint_timeslot]) + total_runtime))
            new_survival_functions = np.concatenate((new_survival_functions, np.array(survival_functions[alg_id][index_event_probabilities:index_endpoint_timeslot])))

        if index_event_probabilities + len(event_times[alg_id]) < index_endpoint_timeslot:
            end_of_timeslot_probability = survival_functions[alg_id][index_endpoint_timeslot-1]

        # Remove the survival curve values that were already used 
        event_times[alg_id] = np.array(event_times[alg_id][index_endpoint_timeslot:]) - timeslot # This might be optimized considering time complexity
        survival_functions[alg_id] = np.array(survival_functions[alg_id][index_endpoint_timeslot:])  # This might be optimized considering time complexity

        total_runtime = end_of_timeslot

    return new_event_times, new_survival_functions


def compute_mean_runtime(event_times: List[float], survival_functions: List[float], cutoff: float, PAR_K: int = 1) -> float:
    '''
    This method computes the mean runtime of a given algorithm or schedule. Increasing the PAR_k parameter the penalty
    for a higher probability in the end also increases.
    '''
    mean_runtime, runtime, scale_probability = 0, 0, 1
    for event_time, probability in zip(event_times, survival_functions):
        timeslot = event_time - runtime
        mean_runtime += scale_probability * timeslot
        runtime, scale_probability = event_time, probability

    mean_runtime += scale_probability * (cutoff - event_time)
    return mean_runtime + (PAR_K - 1) * scale_probability * cutoff


def termination_curve_from_train_data(performance_data: pd.DataFrame, cutoff: float):
    '''
    Given a train data this function computes curves that reflects on how many problem instances
    an algorithm terminated at different times.
    '''
    event_times, termination_functions = [], []
    for algorithm_name in performance_data:
        # Extract necessary event times lower than the cutoff and only once 
        alg_event_times = sorted(performance_data[algorithm_name].values)
        elem_counter = Counter(alg_event_times) #  If it is possible for two algorithms to terminate at the exact same time
        alg_event_times = sorted(set(alg_event_times[:np.searchsorted(alg_event_times, cutoff, 'right')]))
        event_times.append(alg_event_times)

        # Create function values and append them to results
        alg_termination_function, func_value, termination_step = [], 1, 1/len(performance_data)
        for event_time in alg_event_times:
            func_value -= elem_counter[event_time] * termination_step
            alg_termination_function.append(func_value)
        termination_functions.append(alg_termination_function)

    return event_times, termination_functions


def schedule_termination_curve(schedule, cutoff, scenario):
    scenario_dict = scenario.performance_data.T.to_dict()
    schedule_performance_data = pd.DataFrame({'Schedule' : {instance_name: ground_truth_evaluation(schedule, {i:runtime for i, runtime in enumerate(values.values())}, cutoff, 10 * cutoff) for instance_name, values in scenario_dict.items()}})
    return termination_curve_from_train_data(schedule_performance_data, cutoff)


def normalized_par_k_values(value: float, sbs_value: float, vbs_value: float):
    return (value - vbs_value) / (sbs_value - vbs_value)
