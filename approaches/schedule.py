from aslib_scenario.aslib_scenario import ASlibScenario
import logging

from schedule import schedule_termination_curve, compute_mean_runtime
from optimization import run_optimization, solution_to_schedule
import optimization as survopt


class Schedule:
    def __init__(self):
        self.logger = logging.getLogger("schedule")
        self.logger.addHandler(logging.StreamHandler())

        self.performance_value = None

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int, par_k: float = 10):
        pass

    def predict(self, features_of_test_instance, instance_id: int):
        return self.performance_value[instance_id]

    def get_name(self):
        return "schedule"


class SingleBestSchedule:

    def __init__(self):
        self.logger = logging.getLogger("sbschedule")
        self.logger.addHandler(logging.StreamHandler())

        self.performance_value = None

    def fit(self, train_scenario: ASlibScenario, fold: int, amount_of_training_instances: int, par_k: float = 10): # test_scenario: ASlibScenario, 
        pass
        # survopt.SCENARIO = train_scenario
        # survopt.CUTOFF = train_scenario.algorithm_cutoff_time
        # survopt.PAR_K = par_k
        # sbschedule_solution = run_optimization(amount_of_training_instances, train_scenario.algorithm_cutoff_time, max_num_iteration=250, survival=False)
        # sbschedule = solution_to_schedule(sbschedule_solution)
        # static_schedule_event_times, static_schedule_termination_values = schedule_termination_curve(sbschedule, train_scenario.algorithm_cutoff_time, test_scenario)
        # self.performance_value = compute_mean_runtime(static_schedule_event_times[0], static_schedule_termination_values[0], train_scenario.algorithm_cutoff_time, par_k)

    def predict(self, features_of_test_instance, instance_id: int):
        return self.performance_value

    def get_name(self):
        return "sbschedule"


class NPARK_SBAlgorithm:
    def __init__(self):
        self.logger = logging.getLogger("nPARK_SBAlgorithm")
        self.logger.addHandler(logging.StreamHandler())

        self.performance_value = None

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int, par_k: float = 10):
        pass

    def predict(self, features_of_test_instance, instance_id: int):
        return self.performance_value

    def get_name(self):
        return "nPARK_SBAlgorithm"
    

class NPARK_SBSchedule:
    def __init__(self):
        self.logger = logging.getLogger("nPARK_SBSchedule")
        self.logger.addHandler(logging.StreamHandler())

        self.performance_value = None

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int, par_k: float = 10):
        pass

    def predict(self, features_of_test_instance, instance_id: int):
        return self.performance_value

    def get_name(self):
        return "nPARK_SBSchedule"
    

class NPARK_SBSchedule_to_SBAlgorithm:
    def __init__(self):
        self.logger = logging.getLogger("nPARK_SBSchedule_to_SBAlgorithm")
        self.logger.addHandler(logging.StreamHandler())

        self.performance_value = None

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int, par_k: float = 10):
        pass

    def predict(self, features_of_test_instance, instance_id: int):
        return self.performance_value

    def get_name(self):
        return "nPARK_SBSchedule_to_SBAlgorithm"


class MeanOptimizationTime:
    def __init__(self):
        self.logger = logging.getLogger("mean_optimization_time")
        self.logger.addHandler(logging.StreamHandler())

        self.performance_value = None

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int, par_k: float = 10):
        pass

    def predict(self, features_of_test_instance, instance_id: int):
        return self.performance_value

    def get_name(self):
        return "mean_optimization_time"
    

class UnsolvedInstances:

    def __init__(self):
        self.logger = logging.getLogger("unsolved_instances")
        self.logger.addHandler(logging.StreamHandler())

        self.performance_value = None

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int, par_k: float = 10):
        pass

    def predict(self, features_of_test_instance, instance_id: int):
        return self.performance_value

    def get_name(self):
        return "unsolved_instances"