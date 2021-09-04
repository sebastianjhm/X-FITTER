import math
from scipy.optimize import least_squares
import scipy.stats

class LOGARITHMIC:
    """
    Logarithmic distribution
    https://en.wikipedia.org/wiki/Geometric_distribution
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.p = self.parameters["p"]
                
    def cdf(self, x):
        """
        Cumulative distribution function.
        Calculated with quadrature integration method of scipy.
        """
        result = scipy.stats.logser.cdf(x, self.p)
        return result

    
    def pdf(self, x):
        """
        Probability density function
        """
        result = -(self.p**x)/(math.log(1-self.p)*x)
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
        v1 = self.p > 0 and self.p < 1
        return v1

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
            {"alpha": *, "beta": *, "gamma": *}
        """
        def equations(sol_i, measurements):
            ## Variables declaration
            p = sol_i
            
            ## Parametric expected expressions
            parametric_mean = -p/((1-p)*math.log(1-p))
            
            ## System Equations
            eq1 = parametric_mean - measurements.mean
            return (eq1)
        solution = least_squares(equations, 0.5, bounds = (0, 1), args=([measurements]))
        parameters = {"p": solution.x[0]}
        return parameters

if __name__ == '__main__':
    ## Import function to get measurements
    from measurements.measurements import MEASUREMENTS

    ## Import function to get measurements
    def get_data(direction):
        file  = open(direction,'r')
        data = [int(x) for x in file.read().splitlines()]
        return data

    ## Distribution class
    path = "../data/data_logarithmic.txt"
    data = get_data(path)
    measurements = MEASUREMENTS(data)
    distribution = LOGARITHMIC(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(round(measurements.mean)))
    print(distribution.pdf(round(measurements.mean)))