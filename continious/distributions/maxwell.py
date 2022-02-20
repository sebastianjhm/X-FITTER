import scipy.integrate
import math
import scipy.stats
import scipy.special as sc
import numpy as np

class MAXWELL:
    """
    Maxwell distribution
    https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution#:~:text=Mathematically%2C%20the%20Maxwell%E2%80%93Boltzmann%20distribution,of%20temperature%20and%20particle%20mass).  
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha = self.parameters["alpha"]
        self.loc = self.parameters["loc"]
       
        
    def cdf(self, x):
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        # result = scipy.stats.maxwell.cdf(x, loc = self.loc, scale = self.alpha)
        z = lambda t: (x - self.loc) / self.alpha
        result = sc.erf(z(x)/(math.sqrt(2))) - math.sqrt(2 / math.pi) * z(x) * math.exp(-z(x)**2/2)
        return result
    
    def pdf(self, x):
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        # result = scipy.stats.maxwell.pdf(x, loc = self.loc, scale = self.alpha)
        z = lambda t: (x - self.loc) / self.alpha
        result = 1 / self.alpha * math.sqrt(2 / math.pi) * z(x)**2 * math.exp(-z(x)**2/2)
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
        v1 = self.lambda_ > 0
        v2 = self.n1 > 0
        v3 = self.n2 > 0
        return v1 and v2 and v3

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
            {"df": *}
        """
        # def equations(sol_i, measurements):
        #     alpha, loc = sol_i
            
        #     ## Parametric expected expressions
        #     parametric_mean = loc + 2 * alpha * math.sqrt(2 / math.pi)
        #     parametric_variance = alpha ** 2 * (3 * math.pi - 8) / math.pi
        #     parametric_median = loc + alpha * math.sqrt(2 * sc.gammaincinv(1.5, 0.5))
        #     # parametric_mode = loc + sigma * alpha * math.sqrt(2)
            
        #     ## System Equations
        #     eq1 = parametric_mean - measurements.mean
        #     eq2 = parametric_variance - measurements.variance
        #     # eq3 = parametric_mode  - measurements.mode
        #     eq3 = parametric_median - measurements.median
            
        #     return (eq1, eq2, eq3)
        
        # bnds = ((0, -np.inf), (np.inf, np.inf))
        # x0 = (1, measurements.mean)
        # args = ([measurements])
        # solution = scipy.optimize.least_squares(equations, x0, bounds = bnds, args=args)
        # parameters = {"alpha": solution.x[0], "loc": solution.x[1]}
        
        alpha = math.sqrt(measurements.variance * math.pi / (3 * math.pi - 8))
        loc = measurements.mean - 2 * alpha * math.sqrt(2 / math.pi)
        parameters = {"alpha": alpha, "loc": loc}
        
        return parameters

if __name__ == '__main__':
    ## Import function to get measurements
    from measurements_cont.measurements import MEASUREMENTS

    ## Import function to get measurements
    def get_data(direction):
        file  = open(direction,'r')
        data = [float(x.replace(",",".")) for x in file.read().splitlines()]
        return data

    ## Distribution class
    path = "../data/data_maxwell.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = MAXWELL(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
    
    # print(scipy.stats.ncf.fit(data))