import scipy.integrate
import math
from scipy.optimize import fsolve

class BETA:
    """
    Beta distribution
    https://www.vosesoftware.com/riskwiki/Beta4distribution.php          
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha_ = self.parameters["alpha"]
        self.beta_ = self.parameters["beta"]
        self.min_ = self.parameters["min"]
        self.max_ = self.parameters["max"]
        
    def cdf(self, x):
        """
        Cumulative distribution function
        Calculated with quadrature integration method of scipy
        """
        result, error = scipy.integrate.quad(self.pdf, self.min_, x)
        return result
    
    def pdf(self, x):
        """
        Probability density function
        """
        z = lambda x: (x - self.min_) / (self.max_ - self.min_)
        return ( 1 / (self.max_ - self.min_)) * ( math.gamma(self.alpha_ + self.beta_) / (math.gamma(self.alpha_) * math.gamma(self.beta_))) * (z(x)**(self.alpha_-1)) * ((1-z(x))**(self.beta_-1))

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
            {"alpha": *, "beta": *, "min": *, "max": *}
        """
        def equations(sol_i, measurements):
            ## Variables declaration
            alpha_, beta_, min_, max_ = sol_i
            
            ## Parametric expected expressions
            parametric_mean = min_ + (alpha_ / ( alpha_ + beta_ )) * (max_ - min_)
            parametric_variance = ((alpha_ * beta_)/((alpha_ + beta_)**2 * (alpha_ + beta_ + 1))) * (max_ - min_)**2
            parametric_skewness = 2 * ((beta_ - alpha_)/(alpha_ + beta_ + 2)) * math.sqrt((alpha_ + beta_ + 1)/(alpha_ * beta_))
            parametric_kurtosis = 3 * (((alpha_ + beta_ + 1)*(2*(alpha_ + beta_)**2 +(alpha_ * beta_)*(alpha_ + beta_ - 6)))/((alpha_ * beta_)*(alpha_ + beta_ + 2)*(alpha_ + beta_ + 3)))
            
            ## System Equations
            eq1 = parametric_mean - measurements["mean"]
            eq2 = parametric_variance - measurements["variance"]
            eq3 = parametric_skewness - measurements["skewness"]
            eq4 = parametric_kurtosis  - measurements["kurtosis"]
            
            return (eq1, eq2, eq3, eq4)
        
        solution =  fsolve(equations, (1, 1, 1, 1), measurements)
        parameters = {"alpha": solution[0], "beta": solution[1], "min": solution[2], "max": solution[3]}
        return parameters