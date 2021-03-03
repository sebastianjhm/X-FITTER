import scipy.stats
import math
from scipy.optimize import fsolve

class FATIGUE_LIFE:
    """
    Fatigue life distribution
    https://www.vosesoftware.com/riskwiki/FatigueLifedistribution.php
    ** Variance: beta**2 * gamma**2 * (1 + 5 * gamma**2/4)
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        
        self.gamma = self.parameters["gamma"]
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
        
    def cdf(self, x):
        """
        Cumulative distribution function.
        Calculated with quadrature integration method of scipy.
        """
        return scipy.stats.fatiguelife.cdf(x, self.gamma, loc=self.alpha, scale=self.beta)
    
    def pdf(self, x):
        """
        Probability density function
        """
        return scipy.stats.fatiguelife.pdf(x, self.gamma, loc=self.alpha, scale=self.beta)
    
    def get_num_parameters(self):
        """
        Number of parameters of the distribution
        """
        return len(self.parameters.keys())
    
    def parameter_restrictions(self):
        """
        Check parameters restrictions
        """
        v1 = self.beta > 0
        v2 = self.gamma > 0
        return v1 and v2
        
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
            {"gamma": *, "alpha": *, "beta": *}
        """
        ## NO SE ESTÁN RESOLVIENDO LAS ECUACIONES PARA GAMMA = 5, BETA = 10, ALPHA = 5
        # def equations(sol_i, measurements):
        #     ## Variables declaration
        #     alpha, beta, gamma = sol_i
            
        #     ## Parametric expected expressions
        #     parametric_mean = alpha + beta * (1 + gamma**2/2)
        #     parametric_variance = beta**2 * gamma**2 * (1 + 5 * gamma**2/4)
        #     parametric_skewness = 4 * gamma**2 * (11*gamma**2 + 6) / ((4+5*gamma**2)*math.sqrt(gamma**2 * (4+5*gamma**2)))
        
        #     ## System Equations
        #     eq1 = parametric_mean - measurements["mean"]
        #     eq2 = parametric_variance - measurements["variance"]
        #     eq3 = parametric_skewness - measurements["skewness"]
            
        #     return (eq1, eq2, eq3)
        
        # solution =  fsolve(equations, (1, 1, 1), measurements)
        # print(solution)
        scipy_params = scipy.stats.fatiguelife.fit(measurements["data"])
        parameters = {"gamma": scipy_params[0], "alpha": scipy_params[1], "beta": scipy_params[2]}
        return parameters
    
# print(scipy.stats.gamma.ppf(0.48, 24, scale=2))

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

# path = "C:\\Users\\USUARIO1\\Desktop\\Fitter\\data\\data_fatigue_life.txt"
# data = getData(path) 

# print(scipy.stats.fatiguelife.fit(data))
# print(scipy.stats.fatiguelife.cdf(14.783202, 0.2, loc=10, scale=5))

# measurements = get_measurements(data)
# distribution = FATIGUE_LIFE(measurements)
# print(distribution.get_parameters(measurements))
# print(distribution.cdf(5.4455))



# def equations(sol_i, measurements):
#     ## Variables declaration
#     alpha, beta, gamma = sol_i
    
#     ## Parametric expected expressions
#     parametric_mean = alpha + beta * (1 + gamma**2/2)
#     parametric_variance = beta**2 * gamma**2 * (1 + 5 * gamma**2/4)
#     parametric_skewness = 4 * gamma**2 * (11*gamma**2 + 6) / ((4+5*gamma**2)*math.sqrt(gamma**2 * (4+5*gamma**2)))

#     ## System Equations
#     eq1 = parametric_mean - measurements["mean"]
#     eq2 = parametric_variance - measurements["variance"]
#     eq3 = parametric_skewness - measurements["skewness"]
    
#     return (eq1, eq2, eq3)

# solution =  fsolve(equations, (1, 1, 1), measurements)
# print(solution)

# alpha, beta, gamma = 10, 5, 3
# parametric_mean = alpha + beta * (1 + gamma**2/2)
# parametric_variance = beta**2 * gamma**2 * (1 + 5 * gamma**2/4)
# parametric_skewness = 4 * gamma**2 * (11*gamma**2 + 6) / ((4+5*gamma**2)*math.sqrt(gamma**2 * (4+5*gamma**2)))

# print(parametric_mean, parametric_variance, parametric_skewness)