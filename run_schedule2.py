import logging
import sys
import configparser
import multiprocessing as mp

import pandas as pd
import numpy as np
import database_utils
from time import time
from evaluation import evaluate_scenario, print_stats_of_scenario
from approaches.single_best_solver import SingleBestSolver
from approaches.schedule import SingleBestSchedule, Schedule, NPARK_SBAlgorithm, NPARK_SBSchedule, NPARK_SBSchedule_to_SBAlgorithm, MeanOptimizationTime, UnsolvedInstances
from approaches.oracle import Oracle
from approaches.survival_forests.surrogate import SurrogateSurvivalForest
from approaches.survival_forests.auto_surrogate import SurrogateAutoSurvivalForest
from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from baselines.multiclass_algorithm_selector import MultiClassAlgorithmSelector
from sklearn.linear_model import Ridge
from par_10_metric import Par10Metric
from number_unsolved_instances import NumberUnsolvedInstances
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


def initialize_logging():
    logging.basicConfig(filename='logs/log_file.log', filemode='w',
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)


def load_configuration():
    config = configparser.ConfigParser()
    config.read_file(open('conf/experiment_configuration_schedule2.cfg'))
    return config


def print_config(config: configparser.ConfigParser):
    for section in config.sections():
        logger.info(str(section) + ": " + str(dict(config[section])))


def log_result(result):
    logger.info("Finished experiements for scenario: " + result)


def create_approach(approach_names):
    approaches = list()
    for approach_name in approach_names:
        if approach_name == 'schedule':
            approaches.append(Schedule())
        if approach_name == 'sbschedule':
            approaches.append(SingleBestSchedule())
        if approach_name == 'nPARK_SBAlgorithm':
            approaches.append(NPARK_SBAlgorithm())
        if approach_name == 'nPARK_SBSchedule':
            approaches.append(NPARK_SBSchedule())
        if approach_name == 'nPARK_SBSchedule_to_SBAlgorithm':
            approaches.append(NPARK_SBSchedule_to_SBAlgorithm())
        if approach_name == 'mean_optimization_time':
            approaches.append(MeanOptimizationTime())
        if approach_name == 'unsolved_instances': 
            approaches.append(UnsolvedInstances())
        if approach_name == 'sbs':
            approaches.append(SingleBestSolver())
        if approach_name == 'oracle':
            approaches.append(Oracle())
        if approach_name == 'ExpectationSurvivalForest':
            approaches.append(SurrogateSurvivalForest(criterion='Expectation'))
        if approach_name == 'PolynomialSurvivalForest':
            approaches.append(SurrogateSurvivalForest(criterion='Polynomial'))
        if approach_name == 'GridSearchSurvivalForest':
            approaches.append(SurrogateSurvivalForest(criterion='GridSearch'))
        if approach_name == 'ExponentialSurvivalForest':
            approaches.append(SurrogateSurvivalForest(criterion='Exponential'))
        if approach_name == 'SurrogateAutoSurvivalForest':
            approaches.append(SurrogateAutoSurvivalForest())
        if approach_name == 'PAR10SurvivalForest':
            approaches.append(SurrogateSurvivalForest(criterion='PAR10'))
        if approach_name == 'per_algorithm_regressor':
            approaches.append(PerAlgorithmRegressor())
        if approach_name == 'imputed_per_algorithm_rf_regressor':
            approaches.append(PerAlgorithmRegressor(impute_censored=True))
        if approach_name == 'imputed_per_algorithm_ridge_regressor':
            approaches.append(PerAlgorithmRegressor(
                scikit_regressor=Ridge(alpha=1.0), impute_censored=True))
        if approach_name == 'multiclass_algorithm_selector':
            approaches.append(MultiClassAlgorithmSelector())

    return approaches


def schedule_optimization(scenario, fold, approaches, par_k, amount_of_scenario_training_instances, config, tune_hyperparameters):
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

    print("exp mean", np.array(expvals).mean())
    print("opt mean", np.array(optvals).mean())
    # Computing the Single Best Schedule on train data
    survopt.SCENARIO = train_scenario
    survopt.CUTOFF = cutoff
    survopt.PAR_K = par_k
    sbschedule_solution = run_optimization(num_algs, cutoff, max_num_iteration=10000, survival=False)
    sbschedule = solution_to_schedule(survopt.solution_to_runtime_list(sbschedule_solution['variable']), False)
    static_schedule_event_times, static_schedule_termination_values = schedule_termination_curve(sbschedule, cutoff, test_scenario)
    mean_sbschedule_performance = compute_mean_runtime(static_schedule_event_times[0], static_schedule_termination_values[0], cutoff, par_k)

    schedule_lengths_map = dict()
    for sl in schedule_lengths:
        if sl in schedule_lengths_map:
            schedule_lengths_map[sl] = schedule_lengths_map[sl] + 1
        else:
            schedule_lengths_map[sl] = 1

    # Computing performances values and nParK
    mean_schedule_performance = np.mean(schedule_result_on_ground_truths)
    oracle = mean_performance_of_virtualbestsolver(test_scenario, par_k)
    sbs = mean_performance_of_singlebestssolver(train_scenario, test_scenario, par_k)
    nPARK_SBAlgorithm = normalized_par_k_values(mean_schedule_performance, sbs, oracle)
    nPARK_SBSchedule = normalized_par_k_values(mean_schedule_performance, mean_sbschedule_performance, oracle)
    nPARK_SBSchedule_to_SBAlgorithm = normalized_par_k_values(mean_sbschedule_performance, sbs, oracle)

    results = [
        ["oracle", "par10", oracle],
        ["sba", "par10", sbs],
        ["optschedule", "par10", mean_schedule_performance],
        ["sbschedule", "par10", mean_sbschedule_performance],
        ["optschedule", "npar10_sba", nPARK_SBAlgorithm],
        ["optschedule", "npar10_sbs", nPARK_SBSchedule],
        ["sbschedule", "npar10_sba", nPARK_SBSchedule_to_SBAlgorithm],
        ["optschedule", "mean_optimization_time", (optimization_runtime / len(all_event_times))],
        ["optschedule", "unsolved_instances_absolute", len(np.where(np.array(schedule_result_on_ground_truths) >= cutoff)[0])],
        ["optschedule", "unsolved_instances", len(np.where(np.array(schedule_result_on_ground_truths) >= cutoff)[0])/len(all_event_times)],
        ["optschedule", "schedule_length_mean", np.array(schedule_lengths).mean()],
        ["optschedule", "schedule_length_max", np.array(schedule_lengths).max()],
        ["optschedule", "schedule_length_map", str(schedule_lengths_map)],
    ]

    for row in results:
        print(row)

    db_config = load_configuration()
    db_handle, table_name = database_utils.initialize_mysql_db_and_table_name_from_config(db_config)

    sql_statement = "INSERT INTO " + table_name + " (scenario_name, fold, approach, metric, result) VALUES (%s, %s, %s, %s, %s)"
    for res in results:
        db_cursor = db_handle.cursor()
        values = (scenario.scenario, fold, res[0], res[1], str(res[2]))
        db_cursor.execute(sql_statement, values)
        db_handle.commit()
        db_cursor.close()

    db_handle.close()

def func(approach_names, scenario, fold, amount_of_scenario_training_instances, config, tune_hyperparameters):
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    try:
        approaches = create_approach(approach_names)
        par_k = 10  # Currently it will be only optimized on par10
        logger.info(f"Submitted pool task for scenario: {scenario.scenario} on fold {fold}")
        schedule_optimization(scenario, fold, approaches, par_k, amount_of_scenario_training_instances, config, tune_hyperparameters)
    except Exception:
        # todo: look at this exception
        traceback.print_exception(*sys.exc_info())
        print("something went wrong")


#######################
#         MAIN        #
#######################
if __name__ == "__main__":
    initialize_logging()
    config = load_configuration()
    logger.info("Running experiments with config:")
    print_config(config)

    db_handle, table_name = database_utils.initialize_mysql_db_and_table_name_from_config(
        config)
    database_utils.create_table_if_not_exists(db_handle, table_name)

    amount_of_cpus_to_use = int(config['EXPERIMENTS']['amount_of_cpus'])

    scenarios = config["EXPERIMENTS"]["scenarios"].split(",")
    approach_names = config["EXPERIMENTS"]["approaches"].split(",")
    amount_of_scenario_training_instances = int(
        config["EXPERIMENTS"]["amount_of_training_scenario_instances"])
    tune_hyperparameters = bool(int(config["EXPERIMENTS"]["tune_hyperparameters"]))

    for scenario_name in scenarios:
        scenario = ASlibScenario()
        scenario.read_scenario('aslib_data-master/' + scenario_name)
        print_stats_of_scenario(scenario)

        #Parallel(n_jobs=amount_of_cpus_to_use)(delayed(func)(approach_names, scenario, fold, amount_of_scenario_training_instances, config, tune_hyperparameters) for fold in [3, 5, 7])  # range(1, 11))
        for fold in range(3, 4):
            func(approach_names, scenario, fold, amount_of_scenario_training_instances, config, tune_hyperparameters)
        #exit(0)
    logger.info("Finished all experiments.")
