import scipy.stats
import math

class GENERALIZED_EXTREME_VALUE:
    """
    Generalized Extreme Value Distribution
    https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution
    Notation: xi <-> c
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        
        self.c = self.parameters["c"]
        self.miu = self.parameters["miu"]
        self.sigma = self.parameters["sigma"]
        
    def cdf(self, x):
        """
        Cumulative distribution function.
        Calculated with quadrature integration method of scipy.
        """
        z = lambda x: (x - self.miu) / self.sigma
        if self.c == 0:
            return math.exp(-math.exp(-z(x)))
        else:
            return math.exp(-(1+self.c*z(x))**(-1/self.c))
        # return scipy.stats.genectreme.cdf(x, -self.c, loc=self.miu, scale=self.sigma)
    
    def pdf(self, x):
        """
        Probability density function
        """
        z = lambda x: (x - self.miu) / self.sigma
        if self.c == 0:
            return (1/self.sigma) * math.exp(-z(x)-math.exp(-z(x)))
        else:
            return (1/self.sigma) * math.exp(-(1+self.c*z(x))**(-1/self.c)) * (1+self.c*z(x))**(-1-1/self.c)
    
    def get_num_parameters(self):
        """
        Number of parameters of the distribution
        """
        return len(self.parameters.keys())
    
    def parameter_restrictions(self):
        """
        Check parameters restrictions
        """
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, measurements):
        """
        Calculate proper parameters of the distribution from sample measurements.
        The parameters are calculated by formula.
        
        Parameters
        ----------
        measurements : dict
            {"mean": *, "variance": *, "skewness": *, "kurtosis": *, "data": *}

        Returns
        -------
        parameters : dict
            {"c": *, "miu": *, "sigma": *}
        """
        scipy_params = scipy.stats.genextreme.fit(measurements["data"])
        parameters = {"c": -scipy_params[0], "miu": scipy_params[1], "sigma": scipy_params[2]}
        return parameters
    
# def get_measurements(data: list) -> dict:
#     import scipy.stats
#     import numpy as np
#     measurements = {}
    
#     miu_3 = scipy.stats.moment(data, 3)
#     miu_4 = scipy.stats.moment(data, 4)
#     mean = np.mean(data)
#     variance = np.var(data, ddof=1)
#     skewness = miu_3 / pow(np.std(data, ddof=1),3)
#     kurtosis = miu_4 / pow(np.std(data, ddof=1),4)
#     median = np.median(data)
#     mode = scipy.stats.mode(data)[0][0]
    
#     measurements["mean"] = mean
#     measurements["variance"] =  variance
#     measurements["skewness"] = skewness
#     measurements["kurtosis"] = kurtosis
#     measurements["data"] = data
#     measurements["median"] = median
#     measurements["mode"] = mode
    
#     return measurements

    
# def getData(direction):
#     file  = open(direction,'r')
#     data = [float(x.replace(",",".")) for x in file.read().splitlines()]
#     return data

# path = "C:\\Users\\USUARIO1\\Desktop\\Fitter\\data\\data_generalized_extreme_value.txt"
# data = getData(path)

# measurements = get_measurements(data)
# distribution = GENERALIZED_EXTREME_VALUE(measurements)
# print(distribution.get_parameters(measurements))
# print(distribution.cdf(10.57))