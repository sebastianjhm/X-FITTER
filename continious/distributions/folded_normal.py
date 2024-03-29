import numpy as np
import math
import scipy.special as sc
import scipy.optimize

class FOLDED_NORMAL:
    """
    Folded Normal Distribution
    https://en.wikipedia.org/wiki/Folded_normal_distribution          
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
        z1 = lambda t: (t + self.miu) / self.sigma
        z2 = lambda t: (t - self.miu) / self.sigma
        result = 0.5 * (sc.erf(z1(x)/math.sqrt(2)) + sc.erf(z2(x)/math.sqrt(2)))
        return result
    
    def pdf(self, x):
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        result = math.sqrt(2 / (math.pi * self.sigma ** 2)) * math.exp(-(x**2 + self.miu**2)/(2*self.sigma**2)) * math.cosh( self.miu * x / (self.sigma ** 2))
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
        def equations(sol_i, measurements):
            
            ## Variables declaration
            miu, sigma = sol_i
            
            ## Parametric expected expressions
            parametric_mean = sigma * math.sqrt(2 / math.pi) * math.exp(-miu**2/(2*sigma**2)) + miu * sc.erf(miu/math.sqrt(2*sigma**2))
            parametric_variance = miu ** 2 + sigma ** 2 - parametric_mean ** 2
            
            # System Equations
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            
            return (eq1, eq2)
        
        x0 = [measurements.mean, measurements.standard_deviation]
        b = ((-np.inf, 0), (np.inf, np.inf))
        solution = scipy.optimize.least_squares(equations, x0, bounds = b, args=([measurements]))
        parameters = {"miu": solution.x[0], "sigma": solution.x[1]}
        
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
    path = "../data/data_folded_normal.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = FOLDED_NORMAL(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))