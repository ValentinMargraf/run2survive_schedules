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
from schedule import ground_truth_evaluation, normalized_par_k_values, schedule_termination_curve, compute_mean_runtime
from optimization import run_optimization, solution_to_schedule
from evaluation_of_schedules import not_included_indices, mean_performance_of_virtualbestsolver, mean_performance_of_singlebestssolver
import optimization as survopt


logger = logging.getLogger("run")
logger.addHandler(logging.StreamHandler())


def initialize_logging():
    logging.basicConfig(filename='logs/log_file.log', filemode='w',
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)


def load_configuration():
    config = configparser.ConfigParser()
    config.read_file(open('conf/experiment_configuration_schedule.cfg'))
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
    all_event_times, all_survival_functions, cutoff, train_scenario, test_scenario = create_survival_curves(scenario, fold=fold)

    # Removing nonsolvable test instances
    # indices = not_included_indices(test_scenario)
    # all_event_times = list(np.delete(all_event_times, indices, axis=0))
    # all_survival_functions = list(np.delete(all_survival_functions, indices, axis=0))
    # perf_data = np.delete(test_scenario.performance_data.to_numpy(), indices, axis=0)
    # names_of_solvable_problem_instances = np.delete(test_scenario.performance_data.index.to_numpy(), indices)

    # if test_scenario.feature_cost_data is not None:
    #     feature_cost_data = np.delete(test_scenario.feature_cost_data.to_numpy(), indices, axis=0)


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
        solution = run_optimization(num_algs, cutoff, max_num_iteration=200)
        stop_timer = time()
        optimization_runtime += stop_timer - start_timer

        schedule = solution_to_schedule(solution)

        # Evaluating values of the schedule
        gtvalues = pd.DataFrame(test_scenario.performance_data.to_numpy()[instance_id]).to_dict()[0]
        gtvalues_schedule = ground_truth_evaluation(schedule, gtvalues, cutoff, par_k * cutoff)

        # Add feature cost to the runtime
        if test_scenario.feature_cost_data is not None:
            feature_time = test_scenario.feature_cost_data.to_numpy()[instance_id]
            accumulated_feature_time = np.sum(feature_time)
            accumulated_costs = gtvalues_schedule + accumulated_feature_time
        else:
            accumulated_costs = gtvalues_schedule

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

    approaches2 = []
    for approach in approaches:
        approach_name = str(approach.get_name())
        if approach_name == 'schedule':
            approach.performance_value = schedule_result_on_ground_truths
            print(approach.performance_value)
        elif approach_name == 'sbschedule':
            approach.performance_value = mean_sbschedule_performance
            print(approach.performance_value)
        elif approach_name == 'nPARK_SBAlgorithm':
            approach.performance_value = nPARK_SBAlgorithm
            print(approach.performance_value)
        elif approach_name == 'nPARK_SBSchedule':
            approach.performance_value = nPARK_SBSchedule
            print(approach.performance_value)
        elif approach_name == 'nPARK_SBSchedule_to_SBAlgorithm':
            approach.performance_value = nPARK_SBSchedule_to_SBAlgorithm
            print(approach.performance_value)
        elif approach_name == 'mean_optimization_time':
            approach.performance_value = optimization_runtime / number_of_solvable_testinstances
            print(approach.performance_value)
        elif approach_name == 'unsolved_instances': 
            approach.performance_value = len(np.where(np.array(schedule_result_on_ground_truths) >= cutoff)[0])
            print(approach.performance_value)
        approaches2.append(approach)

    # ------- STOP SCHEDULE OPTIMIZATION --------


    if len(approaches2) < 1:
        logger.error("No approaches recognized!")
    for approach in approaches2:
        metrics = list()
        metrics.append(Par10Metric())
        evaluate_scenario((scenario, test_scenario, train_scenario), approach, metrics, amount_of_scenario_training_instances, fold, config, tune_hyperparameters)


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
    pool = mp.Pool(amount_of_cpus_to_use)

    scenarios = config["EXPERIMENTS"]["scenarios"].split(",")
    approach_names = config["EXPERIMENTS"]["approaches"].split(",")
    amount_of_scenario_training_instances = int(
        config["EXPERIMENTS"]["amount_of_training_scenario_instances"])
    tune_hyperparameters = bool(int(config["EXPERIMENTS"]["tune_hyperparameters"]))
    
    for scenario_name in scenarios:
        scenario = ASlibScenario()
        scenario.read_scenario('data/aslib_data-master/' + scenario_name)
        print_stats_of_scenario(scenario)
        
        for fold in range(1, 11):

            approaches = create_approach(approach_names)

            par_k = 10  # Currently it will be only optimized on par10
            logger.info(f"Submitted pool task for scenario: {scenario.scenario} on fold {fold}")
            # pool.apply_async(schedule_optimization, args=(scenario, fold, approaches, par_k, amount_of_scenario_training_instances, config, tune_hyperparameters))
            schedule_optimization(scenario, fold, approaches, par_k, amount_of_scenario_training_instances, config, tune_hyperparameters)

            print(f'Finished evaluation of fold {fold}')

    pool.close()
    pool.join()
    logger.info("Finished all experiments.")
