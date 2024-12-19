import typing as ty

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from numpy import typing as npt

from smartexp import core


# Aca pongo cuales son realmente los tipos
# No hace falta ponerle el mismo nombre pero es comodo
ParamT = npt.NDArray[np.float64]
ControlT = float
OutputT = float

P_NATURE = 1. * np.asarray([1, 2, 3, 4, 5])
DOMAIN = (-100, 100)


def measure(control: ControlT) -> OutputT:
    return np.polynomial.Polynomial(P_NATURE, domain=DOMAIN)(control)


def model(control: ControlT, params: ParamT) -> OutputT:
    return np.polynomial.Polynomial(params, domain=DOMAIN)(control)


class MyFitter(core.Fitter[ParamT, ControlT, OutputT]):

    def fit(self, data: tuple[list[ControlT], list[OutputT]], params0: ParamT) -> tuple[ParamT, ParamT, float]:
        def _internal(params: ParamT):
            c, o = data
            delta = model(np.asarray(c), params) - np.asarray(o)
            return np.sum(delta * delta)

        res = optimize.minimize(_internal, x0=np.zeros_like(P_NATURE))
        return res.x, np.ones_like(res.x), 0

# class MyABCGeneric(core.Fitter[ParamT, ControlT, OutputT]):
#
#     def __init__(self):
#         self.abc = pyabc.ABCSMC(
#             models=model,
#             parameter_priors=self.priors,
#             distance_function=distance,
#             population_size=self.population_size)
#             abc.new("sqlite:////tmp/test.db",
#             dict(data=self.ydata)
#            )
#
#     def distance(simulation, data):
#         delta = data["data"] - simulation["data"]
#         return np.linalg.norm(delta)
#
#     def get_best_value(self, df) -> ParamT:
#         return np.asarray([np.mean(df["p{ndx}"] for ndx in range(5))])
#
#     def get_width_value(self, df) -> ParamT:
#         return np.asarray([np.std(df["p{ndx}"] for ndx in range(5))])
#
#     def get_parameter_size(self, df) -> float:
#         return np.linalg.norm(self.get_width_value(df)/self.get_best_value(df))
#     def fit(self, data: tuple[list[ControlT], list[OutputT]], params0: ParamT) -> tuple[ParamT, ParamT, float]:
#         df, w = self.abc.run().get_distribution()
#         return self.get_best_value(df), self.get_width_value(df), self.get_parameter_size()
#
# class MyABC(MyABCGeneric):
#
#     priors = {
#
#     }
#
#     population_size =50
#

class MyAimer(core.Suggester[ParamT, ControlT, OutputT]):

    def minimizer(self, func: ty.Callable[[ControlT, ], float]) -> ControlT:
        # res = optimize.minimize(func, x0=np.asarray([0]))
        # return res.x
        return np.asarray(np.random.uniform(DOMAIN[0], DOMAIN[1]))


TSE = core.SmartExperiment[ParamT, ControlT, OutputT]

se = TSE(measure, model,
         np.random.random(size=P_NATURE.shape),
         MyFitter, MyAimer, MyFitter)


se.acquire(np.asarray(2.))
print(f"data: {se.data}")
print(f"starting params: {se.last_best_params}")

deltas = []

STEPS = np.arange(1, 10)

for n in STEPS:
    print(f"--- {n:03d} ---")
    se.step()
    print(f"data: {se.data}")
    print(f"params: {se.last_best_params}")
    delta = se.last_best_params - P_NATURE
    deltas.append(np.sum(delta * delta))


plt.plot(STEPS, deltas)
plt.show()