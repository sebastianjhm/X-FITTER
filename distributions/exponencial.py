import math

class EXPONENCIAL:
    """
    Exponential distribution
    https://en.wikipedia.org/wiki/Exponential_distribution         
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.lambda_ = self.parameters["lambda"]
        
    def cdf(self, x):
        """
        Cumulative distribution function.
        Calculated with known formula.
        """
        return 1 - math.e ** (-self.lambda_ * x)
    
    def pdf(self, x):
        """
        Probability density function
        """
        return self.lambda_ * math.e ** (-self.lambda_ * x)
    
    def get_num_parameters(self):
        """
        Number of parameters of the distribution
        """
        return len(self.parameters.keys())
    
    def get_parameters(self, measurements):
        """
        Calculate proper parameters of the distribution from sample measurements.
        The parameters are calculated by solving the equations of the measures expected 
        for this distribution.The number of equations to consider is equal to the number 
        of parameters.
        
        Parameters
        ----------
        measurements : dict
            {"mean": *, "variance": *, "skewness": *, "kurtosis": *, "data": *}

        Returns
        -------
        parameters : dict
            {"lambda": *}
        """
        _mean = measurements["mean"]
        lambda_ = 1 / _mean
        parameters = {"lambda": lambda_}
        return parameters