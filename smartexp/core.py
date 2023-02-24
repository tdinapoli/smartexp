from __future__ import annotations

import abc

import typing as ty


#############################################
# Types use for generic classes and methods
#############################################

ParamT = ty.TypeVar("ParamT")
ControlT = ty.TypeVar("ControlT")
OutputT = ty.TypeVar("OutputT")


ModelT = ty.Callable[[ControlT, ParamT], OutputT]
DataT = ty.Tuple[ty.List[ControlT], ty.List[OutputT]]
#DataT = ty.List[ty.Tuple[ControlT, OutputT]]


class Fitter(ty.Generic[ParamT, ControlT, OutputT], abc.ABC):

    def __init__(self, model: ModelT):
        self.model = model

    @abc.abstractmethod
    def fit(self, data: DataT, params0: ParamT) -> ty.Tuple[ParamT, ParamT, float]:
        pass


FitterBuilderT = ty.Callable[[ModelT], Fitter[ParamT, ControlT, OutputT]]


class Suggester(ty.Generic[ParamT, ControlT, OutputT], abc.ABC):

    fitter: Fitter[ParamT, ControlT, OutputT]

    def __init__(self, 
                 model: ModelT, 
                 fitter_builder: FitterBuilderT,
                 ):
        self.fitter = fitter_builder(model)
    
    @property
    def model(self) -> ModelT:
        return self.fitter.model

    def objective_function(self, control: ControlT, params: ParamT, control_data: ty.List[ControlT], out_data: ty.List[OutputT]) -> float:
        out = self.fitter.model(control, params)
        out_data = out_data.copy()
        control_data.append(control)
        out_data.append(out)
        best_params, ci_params, vol_params = self.fitter.fit((control_data, out_data), params)
        return vol_params

    def minimizer(self, func: ty.Callable[[ControlT, ], float]) -> ControlT:
        raise NotImplementedError

    def suggest_next(self, data: DataT, params: ParamT) -> ControlT:
        control_data, out_data = data

        def _internal(control: ControlT) -> float:
            _control_data = control_data.copy()
            _out_data = out_data.copy()

            out = self.fitter.model(control, params)

            _control_data.append(control)
            _out_data.append(out)

            best_params, ci_params, vol_params = self.fitter.fit((_control_data, _out_data), params)
            return vol_params

        return self.minimizer(_internal)


SuggesterBuilderT = ty.Callable[[ModelT, FitterBuilderT], Suggester[ParamT, ControlT, OutputT]]


class SmartExperiment(ty.Generic[ParamT, ControlT, OutputT]):
    """ An Experiment is a class that can control the experiment instruments and
    make measurements.
    """

    data: DataT

    measure: ty.Callable[[ControlT, ], OutputT]
    model: ModelT

    fitter: Fitter[ParamT, ControlT, OutputT]
    suggester: Suggester[ParamT, ControlT, OutputT]

    last_best_params: ParamT

    def __init__(self: ty.Self, 
                 measure: ty.Callable[[ControlT, ], OutputT],
                 model: ModelT,
                 guess_params: ParamT,
                 fitter_builder: FitterBuilderT,
                 suggester_builder: SuggesterBuilderT,
                 suggester_fitter_builder: FitterBuilderT,
                 ):

        self.last_best_params = guess_params
        self.data = ([], [])

        self.measure = measure
        self.model = model

        self.fitter = fitter_builder(self.model)
        self.suggester = suggester_builder(self.model, suggester_fitter_builder)

    def acquire(self, control: ControlT) -> OutputT:
        out = self.measure(control)
        self.data[0].append(control)
        self.data[1].append(out)
        return out

    def step(self):
        best_params, ci_params, vol_params = self.fitter.fit(self.data, self.last_best_params)
        control = self.suggester.suggest_next(self.data, best_params)
        self.acquire(control)
        self.last_best_params = best_params
