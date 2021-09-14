import scipy.stats
import math
from scipy.optimize import fsolve, least_squares
import numpy as np

class LOGLOGISTIC:
    """
    Loglogistic distribution
    https://en.wikipedia.org/wiki/Log-logistic_distribution
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
        result = x**self.beta/(self.alpha**self.beta+x**self.beta)
        return result
    
    def pdf(self, x):
        """
        Probability density function
        """
        result = self.beta/self.alpha * (x/self.alpha)**(self.beta-1) / ((1+(x/self.alpha)**self.beta)**2)
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
        def equations(sol_i, data_mean, data_variance, data_skewness):
            α, β = sol_i
        
            E = lambda r: (α**r) * (r*math.pi/β) / math.sin(r*math.pi/β)
            
            parametric_mean = E(1)
            parametric_variance = (E(2) - E(1)**2)
            parametric_skewness = (E(3) - 3*E(2)*E(1) + 2*E(1)**3) / ((E(2)-E(1)**2))**1.5
            parametric_kurtosis = (E(4) - 4 * E(1) * E(3) + 6 * E(1)**2 * E(2) - 3 * E(1)**4)/ ((E(2)-E(1)**2))**2
        
            ## System Equations
            eq1 = parametric_mean - data_mean
            eq2 = parametric_variance - data_variance
            
            return (eq1, eq2)
        
        bnds = ((0, 0), (np.inf, np.inf))
        x0 = (measurements.mean, 1/measurements.variance**0.5)
        args = (measurements.mean, measurements.variance, measurements.skewness)
        solution = least_squares(equations, x0, bounds = bnds, args=args)
        parameters = {"alpha": solution.x[0], "beta": solution.x[1]}

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
    path = "../data/data_loglogistic.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = LOGLOGISTIC(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
    
    print("========= Time parameter estimation analisys ========")
    
    import time
    
    def equations(sol_i, data_mean, data_variance, data_skewness):
        α, β = sol_i
    
        E = lambda r: (α**r) * (r*math.pi/β) / math.sin(r*math.pi/β)
        
        parametric_mean = E(1)
        parametric_variance = (E(2) - E(1)**2)
        parametric_skewness = (E(3) - 3*E(2)*E(1) + 2*E(1)**3) / ((E(2)-E(1)**2))**1.5
        parametric_kurtosis = (E(4) - 4 * E(1) * E(3) + 6 * E(1)**2 * E(2) - 3 * E(1)**4)/ ((E(2)-E(1)**2))**2
    
        ## System Equations
        eq1 = parametric_mean - data_mean
        eq2 = parametric_variance - data_variance
        
        return (eq1, eq2)
    
    ti = time.time()
    bnds = ((0, 0), (np.inf, np.inf))
    x0 = (measurements.mean, 1/measurements.variance**0.5)
    args = (measurements.mean, measurements.variance, measurements.skewness)
    solution = least_squares(equations, x0, bounds = bnds, args=args)
    parameters = {"alpha": solution.x[0], "beta": solution.x[1]}
    print(parameters)
    print("Solve equations time: ", time.time() -ti)
    
    ti = time.time()
    scipy_params = scipy.stats.fisk.fit(data)
    parameters = {"alpha": scipy_params[2], "beta": scipy_params[0]}
    print(parameters)
    print("Scipy time get parameters: ",time.time() -ti)