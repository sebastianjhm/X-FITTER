import math
import scipy.special as sc
import scipy.stats
import numpy as np

class NAKAGAMI:
    """
    Nakagami distribution
    https://en.wikipedia.org/wiki/Nakagami_distribution     
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.m = self.parameters["m"]
        self.omega = self.parameters["omega"]
        
    def cdf(self, x):
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        result = sc.gammainc(self.m, (self.m/self.omega) * x**2)
        return result
    
    def pdf(self, x):
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        return (2 * self.m**self.m)/(math.gamma(self.m) * self.omega**self.m) * (x**(2*self.m-1) * math.exp(-(self.m/self.omega) * x**2))
    
    def get_num_parameters(self):
        """
        Number of parameters of the distribution
        """
        return len(self.parameters.keys())
    
    def parameter_restrictions(self):
        """
        Check parameters restriction
        """
        v1 = self.m >= 0.5
        v2 = self.omega > 0
        return v1 and v2

    def get_parameters(self, measurements):
        """
        Calculate proper parameters of the distribution from sample measurements.
        The parameters are calculated by formula.
        
        Parameters
        ----------
        measurements : dict
            {"mean": *, "variance": *, "skewness": *, "kurtosis": *, "median": *, "mode": *}

        Returns
        -------
        parameters : dict
            {"m": *, "omega": *}
        """
        d = np.array(measurements.data)
        
        E_x2 = sum(d*d) / len(d)
        E_x4 = sum(d*d*d*d) / len(d)
        
        omega = E_x2
        m = E_x2 ** 2 / (E_x4 - E_x2**2)
        parameters = {"m": m , "omega": omega}

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
    path = "../data/data_nakagami.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = NAKAGAMI(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))