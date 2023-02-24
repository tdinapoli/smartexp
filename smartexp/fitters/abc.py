from .. import core
import pyabc
import typing as ty

class ABCFitter(core.Fitter):
    def __init__(self, model: core.ModelT):
        super().__init__(model)

    # Esta bien lo que hicimos? acá params0 debería ser una prior, distinto a core.ParamT 
    def fit(self, data: core.DataT, params0: core.ParamT) -> ty.Tuple[core.ParamT, core.ParamT, float]:
        return super().fit(data, params0)