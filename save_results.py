import pandas as pd
import json
import pickle
import os
from pathlib import Path
from typing import Dict, Any


from schedule import compute_mean_runtime
# from survival_tests.survival_curves import create_scenario, create_survival_curves


def save_optimization_results(solution: Dict[str, Any], scenario_name: str, par_k: float):
    '''
    This function save the solutions of an optimization given a scenario and a par_k values for 
    every test_instance.
    '''
    # Save solution
    with open(Path('solutions.json'), 'r') as read_file:
        current_solution_data: Dict[str, Dict] = json.load(read_file)

    for i in range(len(solution.keys())):
        solution[i]['variable'] = list(solution[i]['variable'])
    current_solution_data.setdefault(scenario_name, {})
    current_solution_data[scenario_name][f"PAR{par_k}"] = solution

    with open(Path('solutions.json'), 'w') as write_file:
        json.dump(current_solution_data, write_file)


def get_optimization_results(scenario_name: str, par_k: float):
    with open(Path('solutions.json'), 'r') as read_file:
        current_solution_data: Dict[str, Dict] = json.load(read_file)
    return pd.DataFrame(current_solution_data)[scenario_name][f"PAR{par_k}"]


def get_specific_optimization_results(scenario_name: str, par_k: float, label: str = 'function'):
    if label not in ['function', 'variable']:
        raise NotImplementedError('There are values avaible only for "function" or "variable"!')

    with open(Path('solutions.json'), 'r') as read_file:
        current_solution_data: Dict[str, Dict] = json.load(read_file)

    all_solutions = pd.DataFrame(current_solution_data)
    return [data[label] for data in all_solutions[scenario_name][f"PAR{par_k}"].values()]


def save_schedule_comparison(scenario_name: str, par_k: float, curve_type: str, scenario_properties):
    if curve_type == "survival":
        event_times, survival_functions, cutoff = scenario_properties  
    else:
        event_times, survival_functions, cutoff, termination_event_times, terination_functions = scenario_properties  
    
    df = pd.DataFrame(
        {f'PAR{par_k}':{
            f'Algorithm {i}':round(compute_mean_runtime(event_times[i], survival_functions[i], cutoff, par_k)) 
            for i in range(len(event_times))
        }}) 

    if curve_type == "survival":
        performance_value_schedule = round(get_specific_optimization_results()[scenario_name][f'PAR{par_k}'])
    else:
        performance_value_schedule = round(compute_mean_runtime(termination_event_times[0], terination_functions[0], cutoff, par_k))
    df.set_value('Schedule', value=performance_value_schedule, col=f'PAR{par_k}')


    path = "outputs/" + scenario_name + "_" + curve_type + "_results.csv"
    file_exists = os.path.exists(path)
    if file_exists:
        with open(path, 'r') as read_file:
            current_table = pd.read_csv(read_file, index_col=0)
        current_table[f'PAR{par_k}'] = df
    else:
        current_table = df

    with open(path, 'w') as write_file:
        current_table.to_csv(write_file)


def get_schedule_comparison(scenario_name: str, curve_type: str):
    with open("outputs/" + scenario_name + "_" + curve_type + "_results.csv", 'r') as read_file:
        current_table = pd.read_csv(read_file, index_col=0)
    return current_table


# def save_all_algorithm_performance(scenario_name: str, solution_df: pd.DataFrame, instance_id: int):
#     PARS = [eval(element[3:]) for element in solution_df.index.to_list()]

#     scenario = create_scenario(scenario_name, filepath='../../survival_tests/results/workspaces/aslib/')
#     EVENT_TIMES, SURVIVAL_FUNCTIONS, CUTOFF = create_survival_curves(scenario, instance_id)

#     results = {f'PAR{par}':{f'Algorithm {i}':round(compute_mean_runtime(EVENT_TIMES[i], SURVIVAL_FUNCTIONS[i], CUTOFF, par)) for i in range(len(EVENT_TIMES))} for par in PARS}
#     table_algorithms = pd.DataFrame(results)

#     objectives = {'Schedule': solution_df[scenario_name].to_dict()}
#     objective_table = pd.DataFrame(objectives).apply(round)

#     result_table = pd.concat([objective_table.T, table_algorithms], sort=True)

#     # Save results 
#     with open("outputs/" + scenario_name + "_survival_all_results.csv", 'w') as write_file:
#         result_table.to_csv(write_file)


def save_survival_curves(array, filename: str, array_type: str = "eventtimes", filepath: str="survival_tests/tests/survdata"):
    with open(Path(f"{filepath}/survival_data_{filename}_{array_type}.txt"), 'wb') as file:
        pickle.dump(array, file)


def read_survival_curves(filename: str, array_type: str = "eventtimes", filepath: str="survival_tests/tests/survdata"):
    with open(Path(f"{filepath}/survival_data_{filename}_{array_type}.txt"), 'rb') as file:
        array = pickle.load(file)
    return array