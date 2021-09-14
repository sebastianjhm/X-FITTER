from scipy.optimize import fsolve, least_squares
import numpy as np
import math
import scipy.stats

class RAYLEIGH:
    """
    Rayleigh distribution
    https://en.wikipedia.org/wiki/Rayleigh_distribution    
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.gamma= self.parameters["gamma"]
        self.sigma = self.parameters["sigma"]
        
    def cdf(self, x):
        """
        Cumulative distribution function
        Calculated with quadrature integration method of scipy
        """
        result = 1 - math.exp(-0.5 * ((x - self.gamma)/self.sigma) ** 2)
        return result
    
    def pdf(self, x):
        """
        Probability density function
        """
        result = ((x - self.gamma)/(self.sigma**2)) * math.exp(-0.5 * ((x - self.gamma)/self.sigma) ** 2)
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
        v1 = self.sigma >= 0
        return v1
    
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
        
        σ = math.sqrt(measurements.variance * 2 / (4-math.pi))
        γ = measurements.mean - σ*math.sqrt(math.pi/2)
        
        parameters = {"gamma": γ, "sigma": σ}
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
    path = "../data/data_rayleigh.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = RAYLEIGH(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
        
    print(scipy.stats.rayleigh.fit(measurements.data))