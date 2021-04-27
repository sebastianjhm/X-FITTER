import math
from scipy.optimize import fsolve, least_squares
import numpy as np
import scipy.stats

class GENERALIZED_GAMMA:
    """
    Generalized Gamma Distribution
    https://en.wikipedia.org/wiki/Generalized_gamma_distribution
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        
        self.a = self.parameters["a"]
        self.d = self.parameters["d"]
        self.p = self.parameters["p"]

        
    def cdf(self, x):
        """
        Cumulative distribution function.
        Calculated with quadrature integration method of scipy.
        """
        # import scipy.integrate
        # result, error = scipy.integrate.quad(self.pdf, 0, x)
        result = scipy.stats.gamma.cdf((x/self.a)**self.p, a=self.d/self.p, scale=1)
        return result
    
    def pdf(self, x):
        """
        Probability density function
        """
        return (self.p/(self.a**self.d)) * (x ** (self.d - 1)) * math.exp(-(x/self.a)**self.p) / math.gamma(self.d/self.p)
    
    def get_num_parameters(self):
        """
        Number of parameters of the distribution
        """
        return len(self.parameters.keys())
    
    def parameter_restrictions(self):
        """
        Check parameters restrictions
        """
        v1 = self.a > 0
        v2 = self.d > 0
        v3 = self.p > 0
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
            {"a": *, "c": *, "miu": *, "sigma": *}
        """
        def equations(sol_i, data_mean, data_variance, data_skewness):
            a, d, p = sol_i
            
            E = lambda r: a**r * (math.gamma((d+r)/p)/math.gamma(d/p))
            
            parametric_mean = E(1)
            parametric_variance = E(2) - E(1)**2
            parametric_skewness = (E(3) - 3*E(2)*E(1) + 2*E(1)**3) / ((E(2)-E(1)**2))**1.5
            # parametric_kurtosis = (E(4) - 4 * E(1) * E(3) + 6 * E(1)**2 * E(2) - 3 * E(1)**4)/ ((E(2)-E(1)**2))**2
        
            ## System Equations
            eq1 = parametric_mean - data_mean
            eq2 = parametric_variance - data_variance
            eq3 = parametric_skewness - data_skewness
        
            return (eq1, eq2, eq3)
        ## fsolve is 100x faster than least square but sometimes return solutions < 0
        solution =  fsolve(equations, (1, 1, 1), (measurements["mean"], measurements["variance"], measurements["skewness"]))
        
        ## If return a perameter < 0 then use least_square with restriction
        if all(x > 0 for x in solution) is False or all(x == 1 for x in solution) is True:
            response = least_squares(equations, (1, 1, 1), bounds = ((0, 0, 0), (np.inf, np.inf, np.inf)), args=(measurements["mean"], measurements["variance"], measurements["skewness"]))
            solution = response.x
        parameters = {"a": solution[0], "d": solution[1], "p": solution[2]}
        return parameters
    
if __name__ == '__main__':
    ## Import function to get measurements
    from measurements.data_measurements import get_measurements

    ## Import function to get measurements
    def get_data(direction):
        file  = open(direction,'r')
        data = [float(x.replace(",",".")) for x in file.read().splitlines()]
        return data
    
    ## Distribution class
    path = "..\\data\\data_generalized_gamma.txt"
    data = get_data(path) 
    measurements = get_measurements(data)
    distribution = GENERALIZED_GAMMA(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements["mean"]))
    print(distribution.cdf(62.5))