import math
from scipy.optimize import least_squares
import numpy as np

class DAGUM:
    """
    Dagum distribution
    https://en.wikipedia.org/wiki/Dagum_distribution
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
        self.p = self.parameters["p"]
        
    def cdf(self, x):
        """
        Cumulative distribution function.
        Calculated with quadrature integration method of scipy.
        """
        return (1 + (x/self.b) ** (-self.a)) ** (-self.p)
    
    def pdf(self, x):
        """
        Probability density function
        """
        return (self.a * self.p / x) * (((x/self.b) ** (self.a*self.p))/((((x/self.b) ** (self.a))+1)**(self.p+1)))
        
    def get_num_parameters(self):
        """
        Number of parameters of the distribution
        """
        return len(self.parameters.keys())
    
    def parameter_restrictions(self):
        """
        Check parameters restrictions
        """
        v1 = self.p > 0
        v2 = self.a > 0
        v3 = self.b > 0
        return v1 and v2 and v3

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
        
        def equations(sol_i, measurements):
            ## Variables declaration
            a, b, p = sol_i
            
            ## Parametric expected expressions
            parametric_mean = b * math.gamma(p+1/a) * math.gamma(1-1/a) / math.gamma(p)
            parametric_variance = ((b**2) / (math.gamma(p) ** 2)) * (math.gamma(p) * math.gamma(p+2/a) * math.gamma(1-2/a) - math.gamma(p+1/a) * math.gamma(p+1/a) * math.gamma(1-1/a) * math.gamma(1-1/a))
            parametric_median = b * ((2**(1/p))-1) ** (-1/a)
            
            ## System Equations
            eq1 = parametric_mean - measurements["mean"]
            eq2 = parametric_variance - measurements["variance"]
            eq3 = parametric_median - measurements["median"]
            
            return (eq1, eq2, eq3)
        
        x0 = [measurements["mean"], measurements["mean"], measurements["mean"]]
        b = ((1e-5, 1e-5, 1e-5), (np.inf, np.inf, np.inf))
        solution = least_squares(equations, x0, bounds = b, args=([measurements]))
        parameters = {"a": solution.x[0], "b": solution.x[1], "p": solution.x[2]}
                
        return parameters
    
if __name__ == '__main__':
    ## Import function to get measurements
    from measurements.data_measurements import get_measurements

    ## Import function to get measurements
    def get_data(direction):
        file  = open(direction,'r')
        data = [float(x.replace(",",".")) for x in file.read().splitlines()]
        return data
    
    ## Distribution class
    path = "..\\data\\data_dagum.txt"
    data = get_data(path) 
    measurements = get_measurements(data)
    distribution = DAGUM(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements["mean"]))
