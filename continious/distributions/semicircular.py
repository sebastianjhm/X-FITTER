import numpy as np
import math
import scipy.stats


class SEMICIRCULAR:
    """
    Semicicrcular Distribution
    https://en.wikipedia.org/wiki/Wigner_semicircle_distribution         
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.loc = self.parameters["loc"]
        self.R = self.parameters["R"]

    def cdf(self, x):
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        z = lambda t: t - self.loc
        result = 0.5 + z(x) * math.sqrt(self.R**2 - z(x)**2) / (math.pi * self.R ** 2) + math.asin(z(x)/self.R) / math.pi
        return result
    
    def pdf(self, x):
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        z = lambda t: t - self.loc
        result = 2 * math.sqrt(self.R**2 - z(x)**2) / (math.pi * self.R ** 2)
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
        v1 = self.R > 0
        return v1

    def get_parameters(self, measurements):
        """
        Calculate proper parameters of the distribution from sample measurements.
        The parameters are calculated by formula.
        
        Parameters
        ----------
        measurements : dict
            {"miu": *, "variance": *, "skewness": *, "kurtosis": *, "data": *}

        Returns
        -------
        parameters : dict
            {"miu": *, "sigma": *}
        """
        loc = measurements.mean
        R = math.sqrt(4 * measurements.variance)
        
        ## Correction from domain -R < x < R
        d1 = (loc - R) - measurements.min
        d2 = measurements.max - (loc + R)
        delta = max(max(d1, 0), max(d2, 0)) + 1e-2
        R = R + delta
        parameters = {"loc": loc, "R": R}
        
        return parameters
    
if __name__ == '__main__':
    ## Import function to get measurements
    from measurements_cont.measurements import MEASUREMENTS

    ## Import function to get measurements
    def get_data(direction):
        file  = open(direction,'r')
        data = [float(x.replace(",",".")) for x in file.read().splitlines()]
        return data
    
    ## Distribution class
    path = "../data/data_semicircular.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = SEMICIRCULAR(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))