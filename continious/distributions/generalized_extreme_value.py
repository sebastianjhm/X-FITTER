import scipy.stats
import math

class GENERALIZED_EXTREME_VALUE:
    """
    Generalized Extreme Value Distribution
    https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        
        self.ξ = self.parameters["ξ"]
        self.miu = self.parameters["miu"]
        self.sigma = self.parameters["sigma"]
        
    def cdf(self, x):
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        z = lambda t: (t - self.miu) / self.sigma
        if self.ξ == 0:
            return math.exp(-math.exp(-z(x)))
        else:
            return math.exp(-(1+self.ξ*z(x))**(-1/self.ξ))
        # return scipy.stats.genectreme.cdf(x, -self.ξ, loc=self.miu, scale=self.sigma)
    
    def pdf(self, x):
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        z = lambda t: (t - self.miu) / self.sigma
        if self.ξ == 0:
            return (1/self.sigma) * math.exp(-z(x)-math.exp(-z(x)))
        else:
            return (1/self.sigma) * math.exp(-(1+self.ξ*z(x))**(-1/self.ξ)) * (1+self.ξ*z(x))**(-1-1/self.ξ)
       
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
            {"mean": *, "variance": *, "skewness": *, "kurtosis": *, "median": *, "mode": *}

        Returns
        -------
        parameters : dict
            {"ξ": *, "miu": *, "sigma": *}
        """
        scipy_params = scipy.stats.genextreme.fit(measurements.data)
        parameters = {"ξ": -scipy_params[0], "miu": scipy_params[1], "sigma": scipy_params[2]}
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
    path = "../data/data_generalized_extreme_value.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = GENERALIZED_EXTREME_VALUE(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))