import scipy.integrate
import math

class CHI_SQUARE:
    """
    Chi Square distribution
    https://en.wikipedia.org/wiki/Chi-square_distribution          
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.k = self.parameters["k"]
        
    def cdf(self, x):
        """
        Cumulative distribution function.
        Calculated with quadrature integration method of scipy.
        """
        result, error = scipy.integrate.quad(self.pdf, 0, x)
        return result
    
    def pdf(self, x):
        """
        Probability density function
        """
        return (1/(2**(self.k/2) * math.gamma(self.k/2))) * (x**((self.k/2)-1)) * (math.e ** (-x/2))
    
    def get_num_parameters(self):
        """
        Number of parameters of the distribution
        """
        return len(self.parameters.keys())
    
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
            {"k": *}
        """
        _mean = measurements["mean"]
        k_ = _mean
        parameters = {"k": k_}
        return parameters