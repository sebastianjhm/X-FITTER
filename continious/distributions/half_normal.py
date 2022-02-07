import numpy as np
import math
import scipy.special as sc
import scipy.optimize

class HALF_NORMAL:
    """
    Half Normal Distribution
    https://en.wikipedia.org/wiki/Half-normal_distribution        
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.miu = self.parameters["miu"]
        self.sigma = self.parameters["sigma"]

    def cdf(self, x):
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        z = lambda t: (t - self.miu) / self.sigma
        result = sc.erf(z(x)/math.sqrt(2))
        return result
    
    def pdf(self, x):
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        z = lambda t: (t - self.miu) / self.sigma
        result = (1/self.sigma) * math.sqrt(2/math.pi) * math.exp(-(z(x)**2)/2)
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
        v1 = self.sigma > 0
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
       
        σ = math.sqrt(measurements.variance / (1 - 2/math.pi))
        μ = measurements.mean - σ * math.sqrt(2) / math.sqrt(math.pi)
        parameters = {"miu": μ, "sigma": σ}
        
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
    path = "../data/data_half_normal.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = HALF_NORMAL(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))