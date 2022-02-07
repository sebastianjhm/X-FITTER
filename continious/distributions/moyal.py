import scipy.special as sc
import scipy.stats
import math
import scipy.optimize
import numpy as np

class MOYAL:
    """
    Moyal distribution
    Hand-book on Statistical Distributions (pag.93) ... Christian Walck
    https://reference.wolfram.com/language/ref/MoyalDistribution.html
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
        # result = result = scipy.stats.moyal.cdf(x, loc = self.miu, scale = self.sigma)
        # result = 1 - sc.gammainc(0.5, math.exp(-z(x))/2)
        result = sc.erfc(math.exp(-0.5 * z(x)) / math.sqrt(2))
        return result
    
    def pdf(self, x):
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        z = lambda t: (t - self.miu) / self.sigma
        # result = scipy.stats.moyal.pdf(x, loc = self.miu, scale = self.sigma)
        result = math.exp(-0.5 * (z(x) + math.exp(-z(x)))) / (self.sigma * math.sqrt(2 * math.pi))
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
        # def equations(sol_i, measurements):
        #     ## Variables declaration
        #     μ, σ = sol_i
            
        #     ## Parametric expected expressions
        #     parametric_mean = μ + σ*(math.log(2) + 0.577215664901532)
        #     parametric_variance = σ * σ * math.pi * math.pi / 2
            
        #     ## System Equations
        #     eq1 = parametric_mean - measurements.mean
        #     eq2 = parametric_variance - measurements.variance
            
        #     return (eq1, eq2)
        
        # bnds = ((-np.inf, 0), (np.inf, np.inf))
        # x0 = (measurements.mean, 1)
        # args = ([measurements])
        # solution = scipy.optimize.least_squares(equations, x0, bounds = bnds, args=args)
        
        σ = math.sqrt(2 * measurements.variance / (math.pi * math.pi))
        μ = measurements.mean - σ*(math.log(2) + 0.577215664901532)
        
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
    path = "../data/data_moyal.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = MOYAL(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))