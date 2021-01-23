import math
from scipy.optimize import fsolve

class WEIBULL:
    """
    Weibull distribution
    https://en.wikipedia.org/wiki/Weibull_distribution        
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
        
    def cdf(self, x):
        """
        Cumulative distribution function.
        Calculated with known formula.
        """
        return 1 - math.e ** (-(x / self.beta) ** self.alpha)
    
    def pdf(self, x):
        """
        Probability density function
        """
        return (self.alpha / self.beta) * ((x / self.beta) ** (self.alpha - 1)) * math.e ** (-(x / self.beta) ** self.alpha)
    
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
            {"alpha": *, "beta": *}
        """
        def equations(sol_i, measurements):
            ## Variables declaration
            alpha_, beta_ = sol_i
            
            ## Parametric expected expressions
            parametric_mean = (beta_/alpha_) * math.gamma(1/alpha_)
            parametric_variance = (beta_**2/alpha_) * (2 * math.gamma(2/alpha_) - (1/alpha_) * math.gamma(1/alpha_)**2)

            ## System Equations
            eq1 = parametric_mean - measurements["mean"]
            eq2 = parametric_variance - measurements["variance"]
            return (eq1, eq2)
        
        solution =  fsolve(equations, (1, 1), measurements)
        parameters = {"alpha": solution[0], "beta": solution[1]}
        return parameters