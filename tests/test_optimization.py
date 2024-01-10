import numpy.testing as npt
from typing import List, Dict
from survival_tests.optimization import solution_to_schedule, fitness_function, run_optimization

CUTOFF = 10
DATA1: Dict[str, List[float]] = {'variable': [0, 0, 10]}
DATA2: Dict[str, List[float]] = {'variable': [2, 8, 4]}
DATA3: Dict[str, List[float]] = {'variable': [5, 0, 5, 9]}

##############################
#### solution_to_schedule ####
##############################
def test_solution_to_schedule_one():
    '''
    Tests if the creation of the schedule from the given the solution returned 
    by the optimizer works if only one algorithm will run.
    '''
    expected_schedule = [(2, -1)]
    schedule = solution_to_schedule(DATA1)
    npt.assert_array_almost_equal(schedule, expected_schedule)


def test_solution_to_schedule_many():
    '''
    Tests if the creation of the schedule from the given the solution returned 
    by the optimizer works if several algorithms will run in different order.
    '''
    expected_schedule = [(0, 2), (2, 4), (1, -1)]
    schedule = solution_to_schedule(DATA2)
    npt.assert_array_almost_equal(schedule, expected_schedule)


def test_solution_to_schedule_same_timestep():
    '''
    Tests if the creation of the schedule from the given the solution returned 
    by the optimizer works if several algorithms will run at the same timestep.
    '''
    expected_schedule = [(0, 5), (2, 5), (3, -1)]
    schedule = solution_to_schedule(DATA3)
    npt.assert_array_almost_equal(schedule, expected_schedule)


##########################
#### fitness_function ####
##########################
def test_fitness_function():
    '''
    
    '''
    pass


##########################
#### run_optimization ####
##########################
def test_run_optimization():
    '''
    
    '''
    pass