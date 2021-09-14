import scipy.integrate
import math
import scipy.special as sc

class ERLANG:
    """
    Erlang distribution
    https://www.vosesoftware.com/riskwiki/Erlangdistribution.php        
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.m = self.parameters["m"]
        self.beta = self.parameters["beta"]
        
    def cdf(self, x):
        """
        Cumulative distribution function.
        Calculated with quadrature integration method of scipy.
        """
        lower_inc_gamma = lambda a, x: sc.gammainc(a, x) * math.gamma(a)
        result = lower_inc_gamma(self.m, x/self.beta)/math.gamma(self.m)
        return result
    
    def pdf(self, x):
        """
        Probability density function
        """
        result = ((self.beta ** -self.m) * (x**(self.m-1)) * math.e ** (-(x / self.beta))) / math.factorial(self.m-1)
        return result
    
    def get_num_parameters(self):
        """
        Number of parameters of the distribution
        """
        return len(self.parameters.keys())
    
    def parameter_restrictions(self):
        """
        Check parameters restriction
        """
        v1 = self.m > 0
        v2 = self.beta > 0
        v3 = type(self.m) == int
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
            {"m": *, "beta": *}
        """        
        m = round(measurements.mean ** 2 / measurements.variance)
        β = measurements.variance / measurements.mean
        parameters = {"m": m , "beta": β}
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
    path = "../data/data_erlang.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = ERLANG(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))