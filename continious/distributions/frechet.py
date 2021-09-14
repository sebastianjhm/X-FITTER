import scipy.stats
import math
from scipy.optimize import fsolve

class FRECHET:
    """
    Fréchet distribution
    Also known as inverse Weibull distribution (Scipy name)
    https://en.wikipedia.org/wiki/Fr%C3%A9chet_distribution
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        
        self.alpha = self.parameters["alpha"]
        self.m = self.parameters["m"]
        self.s = self.parameters["s"]
        
    def cdf(self, x):
        """
        Cumulative distribution function.
        Calculated with quadrature integration method of scipy.
        """
        # return scipy.stats.invweibull.cdf(40.89022608, self.alpha, loc = self.m, scale = self.s)
        return math.exp(-((x-self.m)/self.s)**(-self.alpha))
    
    def pdf(self, x):
        """
        Probability density function
        """
        # print(scipy.stats.invweibull.pdf(40.89022608, self.alpha, loc = self.m, scale = self.s))
        return (self.alpha/self.s) * (((x-self.m)/self.s)**(-1-self.alpha)) * math.exp(-((x-self.m)/self.s)**(-self.alpha))
    
    def get_num_parameters(self):
        """
        Number of parameters of the distribution
        """
        return len(self.parameters.keys())
    
    def parameter_restrictions(self):
        """
        Check parameters restrictions
        """
        v1 = self.alpha >= 0
        v2 = self.s >= 0
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
            {"alpha": *, "m": *, "s": *}
        """
        scipy_params = scipy.stats.invweibull.fit(measurements.data)
        parameters = {"alpha": scipy_params[0], "m": scipy_params[1], "s": scipy_params[2]}
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
    path = "../data/data_frechet.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = FRECHET(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))