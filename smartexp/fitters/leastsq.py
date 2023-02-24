from .. import core
import scipy.optimize as opt
import typing as ty
import numpy as np

class LeastSqFitter(core.Fitter):
    def __init__(self, model: core.ModelT):
        super().__init__(model)

    def fit(self, data: core.DataT, params0: core.ParamT, **kwargs) -> ty.Tuple[core.ParamT, core.ParamT, float]:
        xdata, ydata = data
        popt, pcov = opt.curve_fit(self.model, xdata, ydata, p0=params0)
        print(f"{popt=}")
        return popt, np.sqrt(np.diag(pcov)), 0
    
