import numpy.typing as npt

from abc import ABC, abstractmethod
from typing import Sequence, Callable, Tuple, Optional

# A Model is a callable that takes control_params and unknown_params as input
# and returns the results.
Model = Callable[[any], any]

# A measurement is a tuple pair of control_params and results
Measurement = Tuple[npt.ArrayLike, npt.ArrayLike]

# The data is a sequence of Measurement
Data = Sequence[Measurement]

# A fitting_method is a callable that takes a sequence of measurements
# and other optional parameters, and returns a guess on the experiment's 
# unknown_params
Fitting_method = Callable[[Data, Model, Optional[any]], any]

# An inferrer_method is a callable that takes information about unknown_parameters,
# optional parameters and a model, and returns the guess of the best control_parameters to measure next
Inferrer_method = Callable[[any, Model], any]

class Experiment(ABC):
    """ An Experiment is a class that can control the experiment instruments and
    make measurements.
    """
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def measure(self, *params) -> Measurement:
        pass


class SmartExperiment(ABC):
    """ A SmartExperiment is a class that implements an algorithm that consists in 
    multiple steps. Within each step it tries to guess what point in the parameter 
    space of an Experiment will yield a result that makes the Fitting_method converge
    faster on the unknown_params of the Model of the experiment.
    """
    def __init__(self, model: Model, fitting_method: Fitting_method, inferrer_method: Inferrer_method, experiment: Experiment, data: Data): # Change type to npt.ArrayLike[Measurement] ??
        self.model = model
        self.fitting_method = fitting_method
        self.inferrer_method = inferrer_method
        self.experiment = experiment
        self.data = data

    def step(self):
        self.unknown_params = self.fitting_method(self.data, self.model)
        self.best_control_params = self.inferrer_method(
                                            self.unknown_params,
                                            self.model)
        new_data = self.experiment.measure(self.best_control_params)
        self.append_data(new_data)

    @abstractmethod
    def append_data(self, data):
        pass
    

