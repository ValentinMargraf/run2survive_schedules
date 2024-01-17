import logging
import sys
import configparser
import multiprocessing as mp

import pandas as pd
import numpy as np
from time import time
from evaluation import evaluate_scenario, print_stats_of_scenario
from approaches.schedule import Schedule
from par_10_metric import Par10Metric
from aslib_scenario.aslib_scenario import ASlibScenario

from survival_curves import create_survival_curves
from schedule import ground_truth_evaluation, normalized_par_k_values, schedule_termination_curve, compute_mean_runtime, \
    survival_curve_from_schedule
from optimization import run_optimization, solution_to_schedule
from evaluation_of_schedules import not_included_indices, mean_performance_of_virtualbestsolver, mean_performance_of_singlebestssolver
import optimization as survopt
import traceback


logger = logging.getLogger("run")
logger.addHandler(logging.StreamHandler())




def create_approach(approach_names):
    approaches = list()
    for approach_name in approach_names:
        if approach_name == 'schedule':
            approaches.append(Schedule())
    return approaches


def schedule_optimization(scenario, fold, approaches, par_k, amount_of_scenario_training_instances,  tune_hyperparameters):
    # ------- START SCHEDULE OPTIMIZATION--------
    print("create survival curves")
    all_event_times, all_survival_functions, cutoff, train_scenario, test_scenario = create_survival_curves(scenario, fold=fold)

    # Running the optimization
    num_insts = len(all_event_times)
    num_algs = len(all_event_times[0])
    print(scenario.scenario, "fold = ", fold, "|I| = ", num_insts, "|A| = ", num_algs)

    optimization_runtime = 0
    expvals = []
    optvals = []
    schedule_result_on_ground_truths = []
    schedule_lengths = []

    print("start application phase")
    for instance_id in range(num_insts):
        print("instance ", instance_id, "/", num_insts)
        gtvalues = pd.DataFrame(test_scenario.performance_data.to_numpy()[instance_id]).to_dict()[0]
        distinct_gtvalues = list(set(gtvalues.values()))
        if len(distinct_gtvalues) == 1 and distinct_gtvalues[0] >= cutoff:
            continue

        survopt.EVENT_TIMES = all_event_times[instance_id]
        survopt.SURVIVAL_FUNCTIONS = all_survival_functions[instance_id]
        survopt.CUTOFF = cutoff
        survopt.PAR_K = par_k

        best_mean_index = None
        best_mean = None

        survival_frontier = dict()

        def floor_key(d, key):
            if key in d:
                return key
            viable_keys = [k for k in d if k < key]
            if len(viable_keys) > 0:
                return max(viable_keys)
            else:
                return None

        def ceil_key(d, key):
            if key in d:
                return key
            viable_keys = [k for k in d if k > key]
            if len(viable_keys) > 0:
                return min(viable_keys)
            else:
                return None

        for i in range(len(survopt.EVENT_TIMES)):
            if len(survival_frontier) == 0:
                for t_ix in range(len(all_event_times[instance_id][i])):
                    survival_frontier[all_event_times[instance_id][i][t_ix]] = (i, all_survival_functions[instance_id][i][t_ix])
            else:
                for t_ix in range(len(all_event_times[instance_id][i])):
                    time_step = all_event_times[instance_id][i][t_ix]
                    prob = all_survival_functions[instance_id][i][t_ix]
                    compare_time = floor_key(survival_frontier, time_step)
                    add = False
                    if compare_time is not None and prob <= survival_frontier[compare_time][1]:
                        add = True
                    elif compare_time is None:
                        add = True

                    if add:
                        remove_done = False
                        while not remove_done:
                            next_key = ceil_key(survival_frontier, time_step)
                            if next_key is None or prob > survival_frontier[next_key][1]:
                                remove_done = True
                            else:
                                survival_frontier.pop(next_key)
                        survival_frontier[time_step] = (i, prob)

            alg_mean = compute_mean_runtime(all_event_times[instance_id][i], all_survival_functions[instance_id][i], cutoff, par_k)
            if best_mean is None or alg_mean < best_mean:
                best_mean_index = i
                best_mean = alg_mean
        survopt.BEST_INDEX = best_mean_index

        keys = sorted(list(survival_frontier.keys()))
        for i in range(len(keys)):
            if i > 0 and survival_frontier[keys[i]][1] > survival_frontier[keys[i-1]][1]:
                print(keys[i-1], " ", survival_frontier[keys[i-1]])
                print(keys[i], " ", survival_frontier[keys[i]])
                print("!!! ALERT !!!")

        algorithms = set()
        for k in survival_frontier.keys():
            algorithms.add(survival_frontier[k][0])
        print("Algorithm set: ", algorithms)
        algorithms = list(algorithms)
        survopt.FILTERED_ALGORITHMS = algorithms

        if len(algorithms) > 1:
            start_timer = time()
            solution = run_optimization(len(algorithms), cutoff, max_num_iteration=10000)
            stop_timer = time()
            optimization_runtime += stop_timer - start_timer
            schedule = solution_to_schedule(survopt.solution_to_runtime_list(solution['variable']), False)
        else:
            schedule = [(algorithms[0], -1)]

        print("schedule length", len(schedule))
        schedule_lengths.append(len(schedule))
        # Evaluating values of the schedule
        gtvalues_schedule = ground_truth_evaluation(schedule, gtvalues, cutoff, par_k * cutoff)
        optvals.append(gtvalues_schedule)
        expvals.append(ground_truth_evaluation([(best_mean_index, cutoff)], gtvalues, cutoff, par_k * cutoff))

        # Add feature cost to the runtime
        if test_scenario.feature_cost_data is not None:
            feature_time = test_scenario.feature_cost_data.to_numpy()[instance_id]
            accumulated_feature_time = np.sum(feature_time)
            accumulated_costs = gtvalues_schedule + accumulated_feature_time
        else:
            accumulated_costs = gtvalues_schedule

        schedule_result_on_ground_truths.append(accumulated_costs)


    schedule_lengths_map = dict()
    for sl in schedule_lengths:
        if sl in schedule_lengths_map:
            schedule_lengths_map[sl] = schedule_lengths_map[sl] + 1
        else:
            schedule_lengths_map[sl] = 1

    # Computing performances values and nParK
    mean_schedule_performance = np.mean(schedule_result_on_ground_truths)

    print("DONE")
    print("mean_schedule_performance", mean_schedule_performance)

    results = [["optschedule", "par10", mean_schedule_performance]]
    return results

def func(approach_names, scenario, fold, amount_of_scenario_training_instances, tune_hyperparameters):
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    try:
        approaches = create_approach(approach_names)
        par_k = 10  # Currently it will be only optimized on par10
        results = schedule_optimization(scenario, fold, approaches, par_k, amount_of_scenario_training_instances,  tune_hyperparameters)
        return results
    except Exception:
        # todo: look at this exception
        print("something went wrong")
        return None

#######################
#         MAIN        #
#######################
if __name__ == "__main__":

    amount_of_cpus_to_use = 1
    #scenarios = ["ASP-POTASSCO", "BNSL-2016", "CPMP-2015", "CSP-2010", "GRAPHS-2015", "MAXSAT-PMS-2016", "MAXSAT-WPMS-2016", "MAXSAT12-PMS", "MAXSAT15-PMS-INDU", "MIP-2016", "PROTEUS-2014", "QBF-2011", "QBF-2014", "QBF-2016", "SAT03-16_INDU", "SAT11-HAND", "SAT11-INDU", "SAT11-RAND", "SAT12-ALL", "SAT12-HAND", "SAT12-INDU", "SAT12-RAND", "SAT15-INDU", "TSP-LION2015", "TSP-LION2015-ALGO"]
    scenarios = ["ASP-POTASSCO"]#, "BNSL-2016", "CPMP-2015"]
    approach_names = ["schedule"]
    amount_of_scenario_training_instances = int(-1)
    tune_hyperparameters = bool(int(0))

    for scenario_name in scenarios:
        scenario = ASlibScenario()
        scenario.read_scenario('aslib_data/' + scenario_name)
        print_stats_of_scenario(scenario)
        names = ['scenario_name', 'fold', 'approach', 'metric', 'result']

        for fold in range(1, 11):
            results = func(approach_names, scenario, fold, amount_of_scenario_training_instances, tune_hyperparameters)
            try:
                df = pd.read_csv("df.csv")
            except:

                df = pd.DataFrame(columns=names)

            print("DF before results ", df)

            for res in results:


                values = [scenario.scenario, fold, res[0], res[1], str(res[2])]
                print("VALUE ", values)
                # Create a dictionary from column names and data
                row_dict = dict(zip(names, values))

                # Append the new row to the DataFrame
                df = df.append(row_dict, ignore_index=True)


                df.to_csv("results/r2ss_normalized_par_10.csv", index=False)

        #exit(0)
