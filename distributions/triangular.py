import scipy.integrate
import math
from scipy.optimize import fsolve

def get_measurements(data: list) -> dict:
    import scipy.stats
    import numpy as np
    measurements = {}
    
    miu_3 = scipy.stats.moment(data, 3)
    miu_4 = scipy.stats.moment(data, 4)
    mean = np.mean(data)
    variance = np.var(data, ddof=1)
    skewness = miu_3 / pow(np.std(data, ddof=1),3)
    kurtosis = miu_4 / pow(np.std(data, ddof=1),4)
    median = np.median(data)
    mode = scipy.stats.mode(data)[0][0]
    
    measurements["mean"] = mean
    measurements["variance"] =  variance
    measurements["skewness"] = skewness
    measurements["kurtosis"] = kurtosis
    measurements["data"] = data
    measurements["median"] = median
    measurements["mode"] = mode
    
    return measurements

class TRIANGULAR:
    """
    Triangular distribution
    https://en.wikipedia.org/wiki/Triangular_distribution
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
        self.c = self.parameters["c"]
        
    def cdf(self, x):
        """
        Cumulative distribution function.
        Calculated with quadrature integration method of scipy.
        """
        # if self.a < x and x <= self.c:
        #     return (x - self.a)**2/((self.b - self.a)*(self.c - self.a))
        # elif self.c < x and x <= self.b:
        #     return 1 - ((self.b - x)**2/((self.b - self.a)*(self.b - self.c)))
        # elif self.a <= x:
        #     return 0
        # else:
        #     return 1
        result, error = scipy.integrate.quad(self.pdf, self.a, x)
        return result
    
    def pdf(self, x):
        """
        Probability density function
        """
        if self.a <= x and x < self.c:
            return 2*(x - self.a)/((self.b - self.a)*(self.c - self.a))
        elif x == self.c:
            return 2/(self.b - self.a)
        elif x > self.c and x <= self.b:
            return 2*(self.b - x)/((self.b - self.a)*(self.b - self.c))
        
    def get_num_parameters(self):
        """
        Number of parameters of the distribution
        """
        return len(self.parameters.keys())
    
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
            {"a": * , "b": *, "c": *}
        """
        
        # def equations(sol_i, measurements):
        #     ## Variables declaration
        #     a, b, c = sol_i
        #     print(measurements)
        #     ## Parametric expected expressions
        #     parametric_mean = (a + b + c)/3
        #     parametric_variance = (a**2 + b**2 + c**2 - a*b - a*c - b*c)/18
        #     parametric_skewness = math.sqrt(2) * (a + b - 2*c) * (2*a - b - c) * (a - 2*b +c) / (5*(a**2 + b**2 + c**2 - a*b - a*c - b*c)**(3/2))
            
        #     ## System Equations
        #     eq1 = parametric_mean - measurements["mean"]
        #     eq2 = parametric_variance - measurements["variance"]
        #     eq3 = parametric_skewness - measurements["skewness"]
        #     return (eq1, eq2, eq3)
        
        # solution =  fsolve(equations, (1, 1, 1), measurements)
        
        a = min(measurements["data"]) - 1e-8
        b = max(measurements["data"]) + 1e-8
        c = 3 * measurements["mean"] - a - b
        
        parameters = {"a": a, "b": b, "c": c}
        # print(parameters)
        return parameters
    
# def getData(direction):
#     file  = open(direction,'r')
#     data = [float(x.replace(",",".")) for x in file.read().splitlines()]
#     return data

# path = "C:\\Users\\USUARIO1\\Desktop\\Fitter\\data\\data_triangular.txt"
# data = getData(path) 
# measurements = get_measurements(data)
# distribution = TRIANGULAR(measurements)
