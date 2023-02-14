import numpy as np
from template import Experiment, SmartExperiment
import pyabc

class Poly(Experiment):
    def __init__(self):
        self.P_NATURE = np.asarray([1, 5, 3, 2, 9.])
        self.DOMAIN = (-100, 100)
        pass

    def measure(self, x):
        return (x, np.polynomial.Polynomial(self.P_NATURE, domain=self.DOMAIN)(x))

class PolySmart(SmartExperiment):
    def append_data(self, data):
        pass

def model_abc(x, *coef):
    result = np.polynomial.Polynomial(*coef, domain=(-100, 100))(x) 
    return dict(data=result)

def model(x, *coef):
    print(coef)
    return np.polynomial.Polynomial(coef, domain=(-100, 100))(x)


def fitting_method_abc(data, model, population_size=50,
        minimum_epsilon=0.1, max_nr_populations=5):
    x = np.asarray([measurement[0] for measurement in data])
    y = np.asarray([measurement[1] for measurement in data])

    parameter_prior = pyabc.Distribution(
        p0=pyabc.RV("uniform", 0, 10),
        p1=pyabc.RV("uniform", 0, 10),
        p2=pyabc.RV("uniform", 0, 10),
        p3=pyabc.RV("uniform", 0, 10),
        p4=pyabc.RV("uniform", 0, 10)
    )

    def distance(simulation, data):
        delta = data["data"] - simulation["data"]
        return np.linalg.norm(delta)

    abc = pyabc.ABCSMC(
            models=model,
            parameter_priors=parameter_prior,
            distance_function=distance,
            population_size=population_size)
    abc.new("sqlite:////tmp/test.db", 
            dict(data=ydata)
           )
    df, w = abc.run().get_distribution()
    return [np.mean(df["p{ndx}"] for ndx in range(5))]

def fitting_method(data, model, sample_size=50):
    x, y = data[0]
    def distance(sim_data, exp_data):
        delta = abs(sim_data - exp_data)
        return delta

    min_dist = np.inf
    for i in range(sample_size):
        params = np.random.uniform(0.0, 10.0, size=5)
        sim_y = model(x, *params)
        dist = distance(sim_y, y)
        if dist < min_dist:
            min_dist = dist
            parameters = params
    return parameters

def inferrer_method(params, model):
    return np.random.uniform(-100, 100)




experiment = Poly()
measurement0 = experiment.measure(1.0)
data = np.asarray([measurement0])

smart_experiment = PolySmart(model, fitting_method, inferrer_method, experiment, data)

smart_experiment.step()



