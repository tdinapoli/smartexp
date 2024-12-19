from __future__ import annotations

from typing import Callable

class SmartExperiment[DataT, ParamT, ControlT, OutputT]:
    """ An Experiment is a class that can control the experiment instruments and
    make measurements.
    """

    # Naming 
    #   - DataT: a collection of ControlT and resulting DataT
    #   - OutputT: results from a single experimental measurement.
    #   - ControlT: experimental control parameters.
    #   - ParamT: model parameters that should be fitted.

    type SimMeasure = Callable[[ControlT, ParamT], OutputT]

    #: Full dataset.
    data: DataT

    #: Obtain measurement results for a given set of experimental control values.
    measure: Callable[[ControlT, ], OutputT]

    #: Obtain simulated measurement results for a given set of experimental control and model parameter values.
    simulate_measure: SimMeasure

    #: Update the full dataset with the a new experimental control and output.
    #: The third argument is given to enable inplace adding and None is used
    #: for the first call.
    update: Callable[[ControlT, OutputT, DataT], DataT]

    #: Fit full dataset, returning the best parameters with the corresponding uncertainty
    #: or None if the fit was unsuccesful.
    fit: Callable[[SimMeasure, DataT, ParamT], tuple[ParamT, ParamT] | None]

    #: Suggest the new control parameters to use in the measurement.
    suggest: Callable[[SimMeasure, DataT, ParamT, ParamT], ControlT]

    last_best_params: ParamT

    def __init__(self, p0: ParamT):
        self.last_best_params = p0

    def acquire(self, control: ControlT) -> OutputT:
        out = self.measure(control)
        self.data = self.update(control, out, self.data)
        return out

    def step(self) -> bool:
        match self.fit(self.simulate_measure, self.data, self.last_best_params):
            case None:
                return False
            case (best_params, ci_params):
                control = self.suggest(self.simulate_measure, self.data, best_params, ci_params)
                self.acquire(control)
                self.last_best_params = best_params
                return True


class Base:

    def measure(self):
        pass

    def fit(self, input):
        pass

    def process(self, output):
        pass

    def step(self):
        x = self.measure()
        y = self.fit(x)
        return self.process(y)


def step(obj)
    x = obj.measure()
    y = obj.fit(x)
    return obj.process(y)
