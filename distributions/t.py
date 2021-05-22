import scipy.special as sc
import math
import scipy.stats
import numpy as np
from scipy.optimize import least_squares

class T:
    """
    T distribution
    https://en.wikipedia.org/wiki/Student%27s_t-distribution     
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.v = self.parameters["v"]

    def cdf(self, x):
        """
        Cumulative distribution function
        Calculated with quadrature integration method of scipy
        """
        result = scipy.stats.t.cdf(x, self.v)
        return result
    
    def pdf(self, x):
        """
        Probability density function
        """
        result = scipy.stats.t.pdf(x, self.v)
        return result

    def get_num_parameters(self):
        """
        Number of parameters of the distribution
        """
        return len(self.parameters.keys())
    
    def parameter_restrictions(self):
        """
        Check parameters restrictions
        """
        v1 = self.v >= 0
        v2 = type(self.v) == int
        return v1 and v2
    
    def get_parameters(self, measurements):
        """
        Calculate proper parameters of the distribution from sample measurements.
        The parameters are calculated by solving the equations of the measures expected 
        for this distribution.The number of equations to consider is equal to the number 
        of parameters.
        
        Parameters
        ----------
        measurements : dict
            {"mean": *, "variance": *, "skewness": *, "kurtosis": *, "data": *}

        Returns
        -------
        parameters : dict
            {"alpha": *, "beta": *, "min": *, "max": *}
        """
        
        v = 2 * measurements["variance"] / (measurements["variance"] - 1)
        
        parameters = {"v": v}
        
        
        return parameters


if __name__ == "__main__":   
    ## Import function to get measurements
    from measurements.data_measurements import get_measurements
    
    ## Import function to get measurements
    def get_data(direction):
        file  = open(direction,'r')
        data = [float(x.replace(",",".")) for x in file.read().splitlines()]
        return data
    
    ## Distribution class
    path = "..\\data\\data_t.txt"
    data = get_data(path) 
    measurements = get_measurements(data)
    distribution = T(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements["mean"]))
    print(distribution.pdf(measurements["mean"]))

    print(scipy.stats.t.fit((measurements["data"])))