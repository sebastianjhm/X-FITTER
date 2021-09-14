import scipy.stats
from scipy.optimize import least_squares
import numpy as np

class TRAPEZOIDAL:
    """
    Triangular distribution
    https://en.wikipedia.org/wiki/Triangular_distribution
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
        self.c = self.parameters["c"]
        self.d = self.parameters["d"]
        
    def cdf(self, x):
        """
        Cumulative distribution function.
        Calculated with quadrature integration method of scipy.
        """
        if self.a <= x and x < self.b:
            return (1/(self.d + self.c - self.b - self.a)) * (1/(self.b - self.a)) * (x - self.a)**2
        elif self.b <= x and x < self.c:
            return (1/(self.d + self.c - self.b - self.a)) * (2*x - self.a - self.b)
        elif self.c <= x and x <= self.d:
            return 1 - (1/(self.d + self.c - self.b - self.a)) * (1/(self.d - self.c)) * (self.d - x)**2
    
    def pdf(self, x):
        """
        Probability density function
        """
        if self.a <= x and x < self.b:
            return (2/(self.d + self.c - self.b - self.a)) * ((x - self.a)/(self.b - self.a))
        elif self.b <= x and x < self.c:
            return 2/(self.d + self.c - self.b - self.a)
        elif self.c <= x and x <= self.d:
            return (2/(self.d + self.c - self.b - self.a)) * ((self.d - x)/(self.d - self.c))
        
    def get_num_parameters(self):
        """
        Number of parameters of the distribution
        """
        return len(self.parameters.keys())
    
    def parameter_restrictions(self):
        """
        Check parameters restrictions
        """
        v1 = self.b > self.c
        v2 = self.c > self.a
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
            {"a": * , "b": *, "c": *}
        """
        
        a = measurements.min - 1e-3
        d = measurements.max + 1e-3

        def equations(sol_i, measurements, a, d):
            ## Variables declaration
            b, c  = sol_i
            
            ## Parametric expected expressions
            parametric_mean = (1/(3*(d + c - a - b))) * ((d**3-c**3)/(d-c) - (b**3-a**3)/(b-a))
            parametric_variance = (1/(6*(d + c - a - b))) * ((d**4-c**4)/(d-c) - (b**4-a**4)/(b-a)) - ((1/(3*(d + c - a - b))) * ((d**3-c**3)/(d-c) - (b**3-a**3)/(b-a)))**2

            ## System Equations
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
        
            return (eq1, eq2)
        x0 = [(d+a)*0.25, (d+a)*0.75]
        bnds = ((a, a), (d, d))
        solution = least_squares(equations, x0, bounds = bnds, args=([measurements, a, d]))
        
        parameters = {"a": a, "b": solution.x[0], "c": solution.x[1], "d": d}
        return parameters
    
if __name__ == '__main__':
    ## Import function to get measurements
    from measurements.measurements import MEASUREMENTS

    ## Import function to get measurements
    def get_data(direction):
        file  = open(direction,'r')
        data = [float(x.replace(",",".")) for x in file.read().splitlines()]
        return data
    
    ## Distribution class
    path = "../data/data_trapezoidal.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = TRAPEZOIDAL(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
