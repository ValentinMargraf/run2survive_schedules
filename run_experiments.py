import pandas as pd
import numpy as np
from time import time
from py_experimenter.result_processor import ResultProcessor

from survival_tests.survival_curves import create_scenario, create_survival_curves
from survival_tests.schedule import ground_truth_evaluation, normalized_par_k_values, schedule_termination_curve, compute_mean_runtime
from survival_tests.optimization import run_optimization, solution_to_schedule
from survival_tests.evaluation_of_schedules import not_included_indices, mean_performance_of_virtualbestsolver, mean_performance_of_singlebestssolver
import survival_tests.optimization as survopt


def run_experiment(keyfields: dict, result_processor: ResultProcessor, custom_fields: dict):
    '''
    
    '''
    # Extracting given parameters
    scenario_name = keyfields['scenario']
    fold = keyfields['fold']
    par_k = keyfields['par_k']

    # Creating scenarios and survival curves 
    scenario = create_scenario(scenario_name)
    all_event_times, all_survival_functions, cutoff, train_scenario, test_scenario = create_survival_curves(scenario, fold=fold)

    # Removing nonsolvable test instances
    indices = not_included_indices(test_scenario)
    all_event_times = list(np.delete(all_event_times, indices, axis=0))
    all_survival_functions = list(np.delete(all_survival_functions, indices, axis=0))
    perf_data = np.delete(test_scenario.performance_data.to_numpy(), indices, axis=0)
    names_of_solvable_problem_instances = np.delete(test_scenario.performance_data.index.to_numpy(), indices)

    if test_scenario.feature_cost_data is not None:
        feature_cost_data = np.delete(test_scenario.feature_cost_data.to_numpy(), indices, axis=0)


    # Running the optimization
    num_algs = len(all_event_times[0])
    number_of_solvable_testinstances = len(all_event_times)

    optimization_runtime = 0
    schedule_result_on_ground_truths = []
    for instance_id in range(number_of_solvable_testinstances):
        survopt.EVENT_TIMES = all_event_times[instance_id]
        survopt.SURVIVAL_FUNCTIONS = all_survival_functions[instance_id]
        survopt.CUTOFF = cutoff
        survopt.PAR_K = par_k

        start_timer = time()
        solution = run_optimization(num_algs, cutoff, max_num_iteration=300)
        stop_timer = time()
        optimization_runtime += stop_timer - start_timer

        schedule = solution_to_schedule(solution)

        # Evaluating values of the schedule
        gtvalues = pd.DataFrame(perf_data[instance_id]).to_dict()[0]
        gtvalues_schedule = ground_truth_evaluation(schedule, gtvalues, cutoff, par_k * cutoff)

        # Add feature cost to the runtime
        if test_scenario.feature_cost_data is not None:
            feature_time = feature_cost_data[instance_id]
            accumulated_feature_time = np.sum(feature_time)
            accumulated_costs = gtvalues_schedule + accumulated_feature_time
        else:
            accumulated_costs = gtvalues_schedule


        # Logging important schedule attributes
        result_processor.process_logs({
            'schedules_attributes': {
                'instance_id': names_of_solvable_problem_instances[instance_id],
                'schedule': str(solution['function']),
                'fitness_value': solution['function'], 
                'ground_truth_value': accumulated_costs
            }
        })

        schedule_result_on_ground_truths.append(accumulated_costs)

    # Computing the Single Best Schedule on train data
    survopt.SCENARIO = train_scenario
    survopt.CUTOFF = cutoff
    survopt.PAR_K = par_k
    sbschedule_solution = run_optimization(num_algs, cutoff, max_num_iteration=250, survival=False)
    sbschedule = solution_to_schedule(sbschedule_solution)
    static_schedule_event_times, static_schedule_termination_values = schedule_termination_curve(sbschedule, cutoff, test_scenario)
    mean_sbschedule_performance = compute_mean_runtime(static_schedule_event_times[0], static_schedule_termination_values[0], cutoff, par_k)

    # Computing performances values and nParK
    mean_schedule_performance = np.mean(schedule_result_on_ground_truths)
    oracle = mean_performance_of_virtualbestsolver(test_scenario, par_k)
    sbs = mean_performance_of_singlebestssolver(train_scenario, test_scenario, par_k)
    nPARK_SBAlgorithm = normalized_par_k_values(mean_schedule_performance, sbs, oracle)
    nPARK_SBSchedule = normalized_par_k_values(mean_schedule_performance, mean_sbschedule_performance, oracle)
    nPARK_SBSchedule_to_SBAlgorithm = normalized_par_k_values(mean_sbschedule_performance, sbs, oracle)

    # Write first part of final results to database
    resultfields = {
        'nPARK_SBAlgorithm': nPARK_SBAlgorithm,
        'nPARK_SBSchedule': nPARK_SBSchedule,
        'nPARK_SBSchedule_to_SBAlgorithm': nPARK_SBSchedule_to_SBAlgorithm,
        'schedule': np.mean(schedule_result_on_ground_truths),
        'sbs': sbs,
        'oracle': oracle,
        'mean_optimization_time': optimization_runtime / number_of_solvable_testinstances,
        'unsolved_instances': len(np.where(np.array(schedule_result_on_ground_truths) >= cutoff)[0]), 
    }
    result_processor.process_results(resultfields)
