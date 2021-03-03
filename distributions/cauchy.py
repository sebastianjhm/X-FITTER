import math
import scipy.stats
import numpy as np
from scipy.optimize import minimize

class CAUCHY:
    """
    Cauchy distribution
    https://en.wikipedia.org/wiki/Cauchy_distribution     
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.x0 = self.parameters["x0"]
        self.gamma = self.parameters["gamma"]
        
    def cdf(self, x):
        """
        Cumulative distribution function.
        Calculated with quadrature integration method of scipy.
        """
        return (1/math.pi) * math.atan(((x - self.x0) / self.gamma)) + (1/2)
    
    def pdf(self, x):
        """
        Probability density function
        """
        return 1/(math.pi * self.gamma * (1 + ((x - self.x0) / self.gamma)**2))
    
    def get_num_parameters(self):
        """
        Number of parameters of the distribution
        """
        return len(self.parameters.keys())
    
    def parameter_restrictions(self):
        """
        Check parameters restrictions
        """
        v1 = self.gamma > 0
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
            {"x0": *, "gamma": *}
        """
        ## First estimation
        x0_ini = measurements["median"]
        q1 = scipy.stats.scoreatpercentile(measurements["data"], 25)
        q3 = scipy.stats.scoreatpercentile(measurements["data"], 75)
        gamma_ini = (q3 - q1)/2
        
        ## Maximum Likelihood Estimation Cauchy distribution        
        def objective(x):
            x0, gamma = x
            return - sum([math.log(1/(math.pi * gamma * (1 + ((d - x0)/gamma)**2))) for d in measurements["data"]])
        solution = minimize(objective, [x0_ini, gamma_ini], method="SLSQP", bounds = [(-np.inf, np.inf),(0,np.inf)])
       
        ## Results
        parameters = {"x0": solution.x[0], "gamma": solution.x[1]}

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

# path = "C:\\Users\\USUARIO1\\Desktop\\Fitter\\data\\data_cauchy.txt"
# data = getData(path) 
# measurements = get_measurements(data)
# distribution = CAUCHY(measurements)
# print(distribution.get_parameters(measurements))
# print(scipy.stats.cauchy.fit(data))
# print(scipy.stats.cauchy.ppf(0.8559, loc = 0, scale=0.5))