import scipy.integrate
import math
import scipy.stats

class GAMMA:
    """
    Gamma distribution
    https://www.vosesoftware.com/riskwiki/Gammadistribution.php        
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
        
    def cdf(self, x):
        """
        Cumulative distribution function.
        Calculated with quadrature integration method of scipy.
        """
        # result, error = scipy.integrate.quad(self.pdf, 0, x)
        # return result
        return scipy.stats.gamma.cdf(x, a=self.alpha, scale=self.beta)
    
    def pdf(self, x):
        """
        Probability density function
        """
        return ((self.beta ** -self.alpha) * (x**(self.alpha-1)) * math.e ** (-(x / self.beta))) / math.gamma(self.alpha)
    
    def get_num_parameters(self):
        """
        Number of parameters of the distribution
        """
        return len(self.parameters.keys())
    
    def parameter_restrictions(self):
        """
        Check parameters restrictions
        """
        v1 = self.alpha > 0
        v2 = self.beta > 0
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
            {"alpha": *, "beta": *}
        """
        mean = measurements["mean"]
        variance = measurements["variance"]
        
        alpha = mean ** 2 / variance
        beta = variance / mean
        parameters = {"alpha": alpha , "beta": beta}
        return parameters