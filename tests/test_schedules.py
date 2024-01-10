import numpy.testing as npt
import numpy as np
import pandas as pd
from survival_tests.schedule import ground_truth_evaluation, survival_curve_from_schedule, compute_mean_runtime, termination_curve_from_train_data
from typing import List, Dict, Tuple


CUTOFF = 10
NOT_TERMINATING = -1
SCHEDULE1: List[Tuple[int, float]] = [(0, 2), (1, 6), (2, -1)]
SCHEDULE1_1: List[Tuple[int, float]] = [(0, 2), (1, 6), (3, 6), (2, -1)]
SCHEDULE2: List[Tuple[int, float]] = [(3, -1)]
SCHEDULE3: List[Tuple[int, float]] = [(2, 5), (1, 6), (2, -1)]

GTV1: Dict[int, float] = {
    0: 1,
    1: 3,
    2: 12,
    3: 12
}
GTV2: Dict[int, float] = {
    0: 3,
    1: 3,
    2: 8,
    3: 8
}
GTV3: Dict[int, float] = {
    0: 3,
    1: 5,
    2: 4,
    3: 4
}
GTV4: Dict[int, float] = {
    0: 3,
    1: 6,
    2: 12,
    3: 12
}

EVENT_TIMES = [
    [1, 3, 5, 7, 9],
    [2, 4, 6, 8],
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    [5]
]

SURVIVAL_FUNCS = [
    [0.95, 0.9, 0.85, 0.8, 0.75],
    [0.8, 0.5, 0.3, 0.1],
    [0.90, 0.81, 0.72, 0.63, 0.54, 0.45, 0.36, 0.27, 0.18],
    [0]
]


#################################
#### ground_truth_evaluation ####
#################################
def test_ground_truth_evaluation_terminates1():
    '''
    Tests if the evaluation of an algorithm schedule given ground truth values (gtv) works
    if the algorithm schedule terminates in the first timeslot.
    '''
    assert ground_truth_evaluation(SCHEDULE1, GTV1, CUTOFF) == 1


def test_ground_truth_evaluation_terminates2():
    '''
    Tests if the evaluation of an algorithm schedule given ground truth values (gtv) works
    if the algorithm schedule terminates in another timeslot.
    '''
    assert ground_truth_evaluation(SCHEDULE1, GTV2, CUTOFF) == 2 + 3


def test_ground_truth_evaluation_terminates3():
    '''
    Tests if the evaluation of an algorithm schedule given ground truth values (gtv) works
    if the algorithm schedule terminates in last timeslot.
    '''
    assert ground_truth_evaluation(SCHEDULE2, GTV2, CUTOFF) == 8
    assert ground_truth_evaluation(SCHEDULE2, GTV3, CUTOFF) == 4


def test_ground_truth_evaluation_terminates_not1():
    '''
    Tests if the evaluation of an algorithm schedule given ground truth values (gtv) works
    if the algorithm schedule does not terminate.
    '''
    assert ground_truth_evaluation(SCHEDULE2, GTV1, CUTOFF) == NOT_TERMINATING
    assert ground_truth_evaluation(SCHEDULE2, GTV4, CUTOFF) == NOT_TERMINATING


def test_ground_truth_evaluation_terminates_not2():
    '''
    Tests if the evaluation of an algorithm schedule given ground truth values (gtv) works
    if the algorithm schedule does not terminate with another schedule.
    '''
    assert ground_truth_evaluation(SCHEDULE1, GTV4, CUTOFF) == NOT_TERMINATING


def test_ground_truth_evaluation_pause_algorithm1():
    '''
    Tests if the evaluation of an algorithm schedule given ground truth values (gtv) works
    if the algorithm schedule contains a pause algorithm.
    '''
    assert ground_truth_evaluation(SCHEDULE3, GTV2, CUTOFF) == 9
    assert ground_truth_evaluation(SCHEDULE3, GTV3, CUTOFF) == 4


def test_ground_truth_evaluation_pause_algorithm_terminates_not1():
    '''
    Tests if the evaluation of an algorithm schedule given ground truth values (gtv) works
    if the algorithm schedule contains a pause algorithm that does not terminate.
    '''
    assert ground_truth_evaluation(SCHEDULE3, GTV1, CUTOFF) == NOT_TERMINATING


def test_ground_truth_evaluation_pause_algorithm_terminates_not2():
    '''
    Tests if the evaluation of an algorithm schedule given ground truth values (gtv) works
    if the algorithm schedule contains a pause algorithm that does not terminate.
    '''
    assert ground_truth_evaluation(SCHEDULE3, GTV4, CUTOFF) == NOT_TERMINATING


def test_ground_truth_evaluation_cutoff():
    '''
    Tests if the evaluation of an algorithm schedule given ground truth values (gtv) works
    if the algorithm schedule terminates exactly at the cutoff.
    '''
    assert ground_truth_evaluation(SCHEDULE1, GTV3, CUTOFF) == CUTOFF


######################################
#### survival_curve_from_schedule ####
######################################
def test_survival_curve_from_schedule_many():
    '''
    Tests if the creation of the schedule-survival-curve works as expected when 
    the schedule contains several algorithms and one is not improving the process.
    It also tests the case the robustness of the function if an algorithm at a later 
    position does not improve the probability within his runtime.
    '''
    expected_event_times = np.array([1, 4, 6])
    expected_survival_functions = np.array([0.95, 0.8, 0.5])
    event_times, survival_functions = survival_curve_from_schedule(
        SCHEDULE1, EVENT_TIMES, SURVIVAL_FUNCS, CUTOFF)
    # npt.assert_array_almost_equal(event_times, expected_event_times)
    npt.assert_array_almost_equal(survival_functions, expected_survival_functions)


def test_survival_curve_from_schedule_one():
    '''
    Tests if the creation of the schedule-survival-curve works as expected when 
    the schedule contains only one algorithm.
    '''
    expected_event_times = np.array(EVENT_TIMES[3])
    expected_survival_functions = np.array(SURVIVAL_FUNCS[3])
    event_times, survival_functions = survival_curve_from_schedule(
        SCHEDULE2, EVENT_TIMES, SURVIVAL_FUNCS, CUTOFF)
    npt.assert_array_almost_equal(event_times, expected_event_times)
    npt.assert_array_almost_equal(
        survival_functions, expected_survival_functions)


def test_survival_curve_from_schedule_pause():
    '''
    Tests if the creation of the schedule-survival-curve works as expected when 
    the schedule contains one algorithm more than one.
    '''
    expected_event_times = np.array([1, 2, 3, 4, 5, 7, 8, 9, 10])
    expected_survival_functions = np.array(SURVIVAL_FUNCS[2])
    event_times, survival_functions = survival_curve_from_schedule(
        SCHEDULE3, EVENT_TIMES, SURVIVAL_FUNCS, CUTOFF)
    npt.assert_array_almost_equal(event_times, expected_event_times)
    npt.assert_array_almost_equal(
        survival_functions, expected_survival_functions)


def test_survival_curve_from_schedule_monoton():
    '''
    Tests if the resulting array fulfill the monotony conditions.
    '''
    event_times, survival_functions = survival_curve_from_schedule(
        SCHEDULE1_1, EVENT_TIMES, SURVIVAL_FUNCS, CUTOFF)
    npt.assert_array_almost_equal(event_times, np.sort(event_times))
    npt.assert_array_almost_equal(survival_functions, np.flip(np.sort(survival_functions)))


def test_survival_curve_from_schedule_many_on_one_timestep():
    '''
    Tests if the creation of the schedule-survival-curve works as expected when 
    the schedule contains many algorithms on the same timestep.
    '''
    expected_event_times = np.array([1, 4, 6])
    expected_survival_functions = np.array([0.95, 0.8, 0.5])
    event_times, survival_functions = survival_curve_from_schedule(
        SCHEDULE1_1, EVENT_TIMES, SURVIVAL_FUNCS, CUTOFF)
    print(event_times, survival_functions)
    npt.assert_array_almost_equal(event_times, expected_event_times)
    npt.assert_array_almost_equal(survival_functions, expected_survival_functions)


##############################
#### compute_mean_runtime ####
##############################
def test_compute_mean_runtime_schedule():
    '''
    Tests if the mean runtime of a algorithm schedule is computed correctly.
    '''
    expected_runtime = 1 + 3 * 0.95 + 2 * 0.8 + 0.5 * 4
    event_times, survival_functions = survival_curve_from_schedule(
        SCHEDULE1, EVENT_TIMES, SURVIVAL_FUNCS, CUTOFF)
    mean_runtime = compute_mean_runtime(event_times, survival_functions, CUTOFF)
    npt.assert_almost_equal(mean_runtime, expected_runtime)


def test_compute_mean_runtime_schedule_par10():
    '''
    Tests if the mean runtime of a algorithm schedule is computed correctly using the
    PAR10 score.
    '''
    expected_runtime = 1 + 3 * 0.95 + 2 * 0.8 + 0.5 * 4 + 9 * 0.5 * CUTOFF
    event_times, survival_functions = survival_curve_from_schedule(
        SCHEDULE1, EVENT_TIMES, SURVIVAL_FUNCS, CUTOFF)
    mean_runtime = compute_mean_runtime(event_times, survival_functions, CUTOFF, PAR_K=10)
    npt.assert_almost_equal(mean_runtime, expected_runtime)


def test_compute_mean_runtime_algorithm():
    '''
    Tests if the mean runtime of a algorithm schedule is computed correctly.
    '''
    expected_runtime = 1 * 1 + 2 * (0.95 + 0.9 + 0.85 + 0.8) + 0.75
    mean_runtime = compute_mean_runtime(EVENT_TIMES[0], SURVIVAL_FUNCS[0], CUTOFF)
    npt.assert_almost_equal(mean_runtime, expected_runtime)


def test_compute_mean_runtime_algorithm_par10():
    '''
    Tests if the mean runtime of a algorithm schedule is computed correctly.
    '''
    expected_runtime = 1 * 1 + 2 * (0.95 + 0.9 + 0.85 + 0.8) + 0.75 + 0.75 * 9 * CUTOFF
    mean_runtime = compute_mean_runtime(EVENT_TIMES[0], SURVIVAL_FUNCS[0], CUTOFF, PAR_K=10)
    npt.assert_almost_equal(mean_runtime, expected_runtime)


termination_times: Dict[str, List[float]] = {
    'a1' : [100, 100, 100, 100, 100],
    'a2' : [1, 2, 3, 4, 5],
    'a3' : [10, 8, 6, 4, 2],
    'a4' : [2, 1, 2, 1, 2],
    'a5' : [7, 3, 2, 100, 6],
}
PERFORMANCE_DATA = pd.DataFrame(termination_times)


###########################################
#### termination_curve_from_train_data ####
###########################################
def test_termination_curve_from_train_data():
    '''
    Tests if the termination curve is build correctly given train data as input.
    This tests include algorithms were no problem instance was solved, ordered termination times,
    several algorithms that terminate at the exact same time and mixed termination points of time.
    '''
    expected_event_times = [[], [1, 2, 3, 4, 5], [2, 4, 6, 8, 10], [1, 2], [2, 3, 6, 7]]
    expected_termination_functions = [[], [0.8, 0.6, 0.4, 0.2, 0], [0.8, 0.6, 0.4, 0.2, 0], [0.6, 0], [0.8, 0.6, 0.4, 0.2]]
    event_times, termination_functions = termination_curve_from_train_data(PERFORMANCE_DATA, CUTOFF)
    for i in range(5):
        npt.assert_array_almost_equal(event_times[i], expected_event_times[i])
        npt.assert_array_almost_equal(termination_functions[i], expected_termination_functions[i])