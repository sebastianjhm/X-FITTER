import scipy.special as sc
import scipy.stats
import math
from scipy.optimize import fsolve, least_squares
import numpy as np

class LEVY:
    """
    Levy distribution
    https://en.wikipedia.org/wiki/L%C3%A9vy_distribution
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.miu = self.parameters["miu"]
        self.c = self.parameters["c"]
        
    def cdf(self, x):
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        y = lambda x: math.sqrt(self.c/((x-self.miu)))    
    
        # result = sc.erfc(y(x)/math.sqrt(2))
        # result = scipy.stats.levy.cdf(x, loc=self.miu, scale=self.c)
        result = 2-2*scipy.stats.norm.cdf(y(x))
        return result
    
    def pdf(self, x):
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        # result = scipy.stats.levy.pdf(x, loc=self.miu, scale=self.c)
        result = math.sqrt(self.c/(2*math.pi)) * math.exp(-self.c/(2*(x-self.miu))) / ((x-self.miu)**1.5)
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
        v1 = self.c > 0
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
            {"miu": *, "c": *}
        """
        # def equations(sol_i, measurements):
        #     ## Variables declaration
        #     miu, c = sol_i
            
        #     ## Parametric expected expressions
        #     parametric_median = miu +  c/(2*(sc.erfcinv(0.5) ** 2))
        #     parametric_mode = miu + c/3
            
        #     ## System Equations
        #     eq1 = parametric_median - measurements.median
        #     eq2 = parametric_mode - measurements.mode
            
        #     return (eq1, eq2)
    
        
        # bnds = ((-np.inf, 0), (np.inf, np.inf))
        # x0 = (1, 1)
        # args = ([measurements])
        # solution = least_squares(equations, x0, bounds = bnds, args=args)
        # print(solution.x)
        # parameters = {"miu": solution.x[0], "c": solution.x[1]}
        
        scipy_params = scipy.stats.levy.fit(measurements.data)
    
        ## Results
        parameters = {"miu": scipy_params[0], "c": scipy_params[1]}

        return parameters

if __name__ == "__main__":   
    ## Import function to get measurements
    from measurements.measurements import MEASUREMENTS
    
    ## Import function to get measurements
    def get_data(direction):
        file  = open(direction,'r')
        data = [float(x.replace(",",".")) for x in file.read().splitlines()]
        return data
    
    ## Distribution class
    path = "../data/data_levy.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = LEVY(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))