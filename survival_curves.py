import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from aslib_scenario.aslib_scenario import ASlibScenario
from sksurv.ensemble import RandomSurvivalForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pathlib import Path
from typing import List, Dict, Any
from save_results import save_survival_curves


def create_scenario(scenario_name: str, filepath: str = 'survival_tests/results/workspaces/aslib/'):
    ''' 
    Load ASlib Scenario. 
    '''
    scenario = ASlibScenario()
    scenario.read_scenario(
        Path(filepath + scenario_name))
    return scenario


def create_survival_curves(scenario: ASlibScenario, instance_id: int=None, fold: int = 1, params: Dict[str, Any] = None, filename:str=None, filepath:str="survival_tests/tests/survdata"):
    test_scenario, train_scenario = scenario.get_split(indx=fold)
    num_algorithms = len(train_scenario.algorithms)
    # num_instances = train_scenario.instances
    algorithm_cutoff_time = train_scenario.algorithm_cutoff_time
    features_train = train_scenario.feature_data.to_numpy()
    performances = train_scenario.performance_data.to_numpy()

    # Might be set as a parameter in a later state of this elaboration
    params = {
        'n_estimators': 100,
        'min_samples_split': 10,
        'min_samples_leaf': 15,
        'min_weight_fraction_leaf': 0.0,
        'max_features': 'sqrt',
        'bootstrap': True,
        'oob_score': False
    }

    models = create_model_structure(num_algorithms, fold, params)
    imputer = [SimpleImputer() for _ in range(num_algorithms)]
    scaler = [StandardScaler() for _ in range(num_algorithms)]

    for alg_id in range(num_algorithms):
        # prepare survival forest dataset and split the data accordingly
        X_train, Y_train = construct_dataset_for_algorithm_id(
            features_train, performances, alg_id, algorithm_cutoff_time)
        X_train = imputer[alg_id].fit_transform(features_train)
        X_train = scaler[alg_id].fit_transform(X_train)
        models[alg_id].fit(X_train, Y_train)

    ##### Predict Survival Functions and respective Risks #####
    all_event_times, all_survival_functions = [], []
    for instance_idx in range(len(test_scenario.feature_data.to_numpy())):
        features_test = test_scenario.feature_data.to_numpy()[instance_idx]
        event_times, survival_functions = [], []
        for alg_id in range(num_algorithms):
            X_test = np.reshape(features_test, (1, -1))
            X_test = imputer[alg_id].transform(X_test)
            X_test = scaler[alg_id].transform(X_test)
            event_times.append(models[alg_id].event_times_)
            survival_functions.append(
                models[alg_id].predict_survival_function(X_test)[0])
        all_event_times.append(event_times)
        all_survival_functions.append(survival_functions)

    if filename:
        save_survival_curves(all_event_times, filename, array_type = "eventtimes", filepath=filepath)
        save_survival_curves(all_survival_functions, filename, array_type = "survivalfunctions", filepath=filepath)

    if instance_id is not None:
        return all_event_times[instance_id], all_survival_functions[instance_id], algorithm_cutoff_time, train_scenario, test_scenario
    return all_event_times, all_survival_functions, algorithm_cutoff_time, train_scenario, test_scenario


def create_model_structure(num_algorithms: int, fold: int, params: Dict[str, Any]) -> List[RandomSurvivalForest]:
    models = [RandomSurvivalForest(n_estimators=params['n_estimators'],
                                   min_samples_split=params['min_samples_split'],
                                   min_samples_leaf=params['min_samples_leaf'],
                                   min_weight_fraction_leaf=params['min_weight_fraction_leaf'],
                                   max_features=params['max_features'],
                                   bootstrap=params['bootstrap'],
                                   oob_score=params['oob_score'],
                                   n_jobs=1,
                                   random_state=fold) for _ in range(num_algorithms)]
    return models


def construct_dataset_for_algorithm_id(instance_features, performances, algorithm_id: int, algorithm_cutoff_time):
    '''
    Fit Random Survival Forest.
    '''
    # get runtimes of algorithm
    performances_of_algorithm_with_id = performances.iloc[:, algorithm_id].to_numpy() if isinstance(performances, pd.DataFrame) else performances[:, algorithm_id]
    num_instances = len(performances_of_algorithm_with_id)

    # for each instance determine whether it was finished before cutoff; also set PAR10 values
    finished_before_timeout = np.empty(num_instances, dtype=bool)
    for i in range(0, len(performances_of_algorithm_with_id)):
        finished_before_timeout[i] = True if (
            performances_of_algorithm_with_id[i] < algorithm_cutoff_time) else False
        if performances_of_algorithm_with_id[i] >= algorithm_cutoff_time:
            performances_of_algorithm_with_id[i] = (algorithm_cutoff_time * 10)

    # for each instance build target, consisting of (censored, runtime)
    status_and_performance_of_algorithm_with_id = np.empty(dtype=[('cens', np.bool), ('time', np.float)],
                                                           shape=instance_features.shape[0])
    status_and_performance_of_algorithm_with_id['cens'] = finished_before_timeout
    status_and_performance_of_algorithm_with_id['time'] = performances_of_algorithm_with_id

    if isinstance(instance_features, pd.DataFrame):
        instance_features = instance_features.to_numpy()

    return instance_features, status_and_performance_of_algorithm_with_id.T


def prepare_survival_curves_for_plot(event_times, survival_functions, algorithm_cutoff_time):
    for alg_id in range(len(event_times)):
        event_times[alg_id] = np.append(0.0, event_times[alg_id])
        event_times[alg_id] = np.append(event_times[alg_id], algorithm_cutoff_time)
        survival_functions[alg_id] = np.append(1.0, survival_functions[alg_id])

        # Repeat last survival probability for plot
        survival_functions[alg_id] = np.append(
            survival_functions[alg_id], survival_functions[alg_id][-1])

    return event_times, survival_functions


def plot_survival_funcs(event_times, survival_functions, algorithm_cutoff_time, save: bool = False, filename: str = "survival_functions"):
    event_times, survival_functions = prepare_survival_curves_for_plot(event_times, survival_functions, algorithm_cutoff_time)
    
    # set figure settings
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.set_ylabel('Survival Probability')
    ax.set_xlabel('Normalized Algorithm Runtime')

    # plot survival functions
    for alg_id in range(len(event_times)):
        events = np.array(event_times[alg_id]) / algorithm_cutoff_time
        ax.step(x=events, y=survival_functions[alg_id],
                where='post', label='Algorithm {}'.format(alg_id))

    ax.legend()
    if save:
        fig.savefig(rf'./{filename}.pdf', bbox_inches='tight')
    fig.show()


def main():
    scenario_name = 'QBF-2011'
    scenario = create_scenario(scenario_name)

    params = {
        'n_estimators': 100,
        'min_samples_split': 10,
        'min_samples_leaf': 15,
        'min_weight_fraction_leaf': 0.0,
        'max_features': 'sqrt',
        'bootstrap': True,
        'oob_score': False
    }
    instance_id = 6

    event_times, survival_functions, algorithm_cutoff_time = create_survival_curves(
        scenario, params, instance_id
    )

    plot_survival_funcs(event_times, survival_functions, algorithm_cutoff_time, True)


if __name__ == "__main__":
    main()
