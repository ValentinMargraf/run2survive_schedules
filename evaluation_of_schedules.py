import numpy as np
from aslib_scenario.aslib_scenario import ASlibScenario
from approaches.single_best_solver import SingleBestSolver


def index_of_singlebestsolver(scenario: ASlibScenario):
    '''
    This function determines the index of the single best solver of a 
    given scenario.
    '''
    sbs = SingleBestSolver()
    sbs.fit(scenario, fold= None, amount_of_training_instances = None)
    return np.argmin(sbs.mean_train_runtimes)


def mean_performance_of_singlebestssolver(train_scenario: ASlibScenario, test_scenario: ASlibScenario, PAR_K: float):
    '''
    This function returns the par_k of the single best solver (sbs) given
    a scenario. 
    '''
    perf_values = test_scenario.performance_data.to_numpy()[:, index_of_singlebestsolver(train_scenario)]
    perf_values = np.delete(perf_values, not_included_indices(test_scenario))
    perf_values[perf_values == test_scenario.algorithm_cutoff_time * 10] = test_scenario.algorithm_cutoff_time * PAR_K
    return np.mean(perf_values)


def not_included_indices(test_scenario: ASlibScenario):
    perf_values = np.min(test_scenario.performance_data.to_numpy(), axis=1)
    current_penalty = 10 * test_scenario.algorithm_cutoff_time
    return np.where(perf_values==current_penalty)


def mean_performance_of_virtualbestsolver(test_scenario: ASlibScenario, PAR_K: float):
    '''
    This function determines the mean value of the best performances of any algorithms on 
    each problem instance given with the scenario which corresponds to the virtual best solver.
    '''
    perf_values = np.min(test_scenario.performance_data.to_numpy(), axis=1)
    perf_values = np.delete(perf_values, not_included_indices(test_scenario))
    perf_values[perf_values == 10 * test_scenario.algorithm_cutoff_time] = test_scenario.algorithm_cutoff_time * PAR_K
    return np.mean(perf_values)


def normalized_par_k(schedule_performance_values, train_scenario: ASlibScenario, test_scenario: ASlibScenario, PAR_K: float = 1):
    '''
    This function computes the normalized par k (nPARK) score from the mean
    performance values of the schedule, singlebestsolver and the virtualbestsolver.
    '''
    vbs = mean_performance_of_virtualbestsolver(test_scenario, PAR_K)
    numerator = np.mean(schedule_performance_values) - vbs
    denominator = mean_performance_of_singlebestssolver(train_scenario, test_scenario, PAR_K) - vbs
    return numerator / denominator