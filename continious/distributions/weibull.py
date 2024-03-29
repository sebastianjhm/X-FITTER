import math
import numpy as np
import scipy.optimize
import scipy.stats

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
        Calculated using definition of the function in the documentation
        """
        return (self.alpha / self.beta) * ((x / self.beta) ** (self.alpha - 1)) * math.e ** (-(x / self.beta) ** self.alpha)
    
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
            {"mean": *, "variance": *, "skewness": *, "kurtosis": *, "median": *, "mode": *}

        Returns
        -------
        parameters : dict
            {"alpha": *, "beta": *}
        """
        def equations(sol_i, measurements):
            ## Variables declaration
            α, β = sol_i
            
            ## Generatred moments function (not-centered)
            E = lambda k: (β**k)*math.gamma(1+k/α)
            
            ## Parametric expected expressions
            parametric_mean = E(1)
            parametric_variance = (E(2) - E(1)**2)
            # parametric_skewness = (E(3) - 3*E(2)*E(1) + 2*E(1)**3) / ((E(2)-E(1)**2))**1.5
            # parametric_kurtosis = (E(4) - 4 * E(1) * E(3) + 6 * E(1)**2 * E(2) - 3 * E(1)**4)/ ((E(2)-E(1)**2))**2
            
            ## System Equations
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            # eq3 = parametric_skewness - measurements.skewness
            # eq4 = parametric_kurtosis  - measurements.kurtosis
            
            return (eq1, eq2)
        
        solution =  scipy.optimize.fsolve(equations, (1, 1), measurements)
        parameters = {"alpha": solution[0], "beta": solution[1]}
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
    path = "../data/data_weibull.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = WEIBULL(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
    
    
    
    print("\n========= Time parameter estimation analisys ========")
    
    import time
    
    def equations(sol_i, measurements):
        ## Variables declaration
        α, β = sol_i
        
        ## Generatred moments function (not-centered)
        E = lambda k: (β**k)/np.prod(np.array([(α-i) for i in range(1,k+1)]))
        
        ## Parametric expected expressions
        parametric_mean = E(1)
        parametric_variance = (E(2) - E(1)**2)
        # parametric_skewness = (E(3) - 3*E(2)*E(1) + 2*E(1)**3) / ((E(2)-E(1)**2))**1.5
        # parametric_kurtosis = (E(4) - 4 * E(1) * E(3) + 6 * E(1)**2 * E(2) - 3 * E(1)**4)/ ((E(2)-E(1)**2))**2
        
        ## System Equations
        eq1 = parametric_mean - measurements.mean
        eq2 = parametric_variance - measurements.variance
        # eq3 = parametric_skewness - measurements.skewness
        # eq4 = parametric_kurtosis  - measurements.kurtosis
        
        return (eq1, eq2)
    
    ti = time.time()
    bnds = ((0, 0), (np.inf, np.inf))
    x0 = (measurements.mean, measurements.mean)
    args = ([measurements])
    solution = scipy.optimize.least_squares(equations, x0, bounds = bnds, args=args)
    parameters = {"alpha": solution.x[0], "beta": solution.x[1]}
    print(parameters)
    print("Solve equations time: ", time.time() -ti)
    
    ti = time.time()
    scipy_params = scipy.stats.weibull_min.fit(measurements.data)
    parameters = {"alpha": scipy_params[0], "beta": scipy_params[2]}
    print(parameters)
    print("Scipy time get parameters: ",time.time() -ti)