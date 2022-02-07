import numpy as np
import scipy.optimize
import scipy.special as sc


class PERT:
    """
    Pert distribution
    https://en.wikipedia.org/wiki/PERT_distribution    
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.min = self.parameters["min"]
        self.max = self.parameters["max"]
        self.mode = self.parameters["mode"]
        
    def cdf(self, x):
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        α1 = (4*self.mode + self.max - 5*self.min) / (self.max - self.min)
        α2 = (5*self.max - self.min - 4*self.mode) / (self.max - self.min)
        z = lambda t: (t - self.min) / (self.max - self.min)
        
        # result = scipy.stats.beta.cdf(z(x), α1, α2)
        result = sc.betainc(α1, α2, z(x))
        
        return result
    
    def pdf(self, x):
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        α1 = (4*self.mode + self.max - 5*self.min) / (self.max - self.min)
        α2 = (5*self.max - self.min - 4*self.mode) / (self.max - self.min)
        return (x-self.min)**(α1-1) * (self.max-x)**(α2-1) / (sc.beta(α1, α2) * (self.max-self.min)**(α1+α2-1))

    def get_num_parameters(self):
        """
        Number of parameters of the distribution
        """
        return len(self.parameters.keys())
    
    def parameter_restrictions(self):
        """
        Check parameters restrictions
        """
        v1 = self.min < self.mode
        v2 = self.mode < self.max
        return v1 and v2
    
    def get_parameters(self, measurements):
        """
        Calculate proper parameters of the distribution from sample measurements.
        The parameters are calculated by solving the equations of the measures expected 
        for this distribution.The number of equations to consider is equal to the number 
        of parameters.
        
        Parameters
        ----------
        measurements : dict
            {"mean": *, "variance": *, "skewness": *, "kurtosis": *, "median": *, "mode": *}

        Returns
        -------
        parameters : dict
            {"alpha": *, "beta": *, "min": *, "max": *}
        """
        def equations(sol_i, measurements):
            min_, max_, mode = sol_i
        
            α1 = (4*mode + max_ - 5*min_) / (max_ - min_)
            α2 = (5*max_ - min_ - 4*mode) / (max_ - min_)
            
            parametric_mean = (min_ + 4*mode + max_)/6
            parametric_variance = ((parametric_mean - min_) * (max_ - parametric_mean)) / 7
            # parametric_skewness = 2 * (α2-α1) * math.sqrt(α2+α1+1) / ((α2+α1+2)* math.sqrt(α2*α1))
            # parametric_kurtosis = 3 + 6*((α2-α1)**2 * (α2+α1+1)-(α2*α1)*(α2+α1+2))/((α2*α1)*(α2+α1+2)*(α2+α1+3))
            #parametric_median = (min_ + 6*mode + max_)/8
            parametric_median = sc.betaincinv(α1, α2, 0.5)*(max_-min_)+min_
            
            ## System Equations
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            # eq3 = parametric_skewness - measurements.skewness
            # eq4 = parametric_kurtosis  - measurements.kurtosis
            eq5 = parametric_median  - measurements.median
            
            return (eq1, eq2, eq5)
        
        bnds = ((-np.inf, measurements.mean, measurements.min), (measurements.mean, np.inf, measurements.max))
        x0 = (measurements.min, measurements.max, measurements.mean)
        args = ([measurements])
        solution = scipy.optimize.least_squares(equations, x0, bounds = bnds, args=args)
        parameters = {"min": solution.x[0], "max": solution.x[1], "mode": solution.x[2]}
        
        parameters["min"] = min(measurements.min-1e-3, parameters["min"])
        parameters["max"] = max(measurements.max+1e-3, parameters["max"])
        
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
    path = "../data/data_pert.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = PERT(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
    




    def equations(sol_i, measurements):
        min_, max_, mode = sol_i
    
        α1 = (4*mode + max_ - 5*min_) / (max_ - min_)
        α2 = (5*max_ - min_ - 4*mode) / (max_ - min_)
        
        parametric_mean = (min_ + 4*mode + max_)/6
        parametric_variance = ((parametric_mean - min_) * (max_ - parametric_mean)) / 7
        # parametric_skewness = 2 * (α2-α1) * math.sqrt(α2+α1+1) / ((α2+α1+2)* math.sqrt(α2*α1))
        # parametric_kurtosis = 3 + 6*((α2-α1)**2 * (α2+α1+1)-(α2*α1)*(α2+α1+2))/((α2*α1)*(α2+α1+2)*(α2+α1+3))
        parametric_median = (min_ + 6*mode + max_)/8
        
        ## System Equations
        eq1 = parametric_mean - measurements.mean
        eq2 = parametric_variance - measurements.variance
        # eq3 = parametric_skewness - measurements.skewness
        # eq4 = parametric_kurtosis  - measurements.kurtosis
        eq5 = parametric_median  - measurements.median
        
        return (eq1, eq2, eq5)
    
    import time
    print("=====")
    ti = time.time()
    bnds = ((-np.inf, measurements.mean, measurements.min), (measurements.mean, np.inf, measurements.max))
    x0 = (measurements.min, measurements.max, measurements.mean)
    args = ([measurements])
    solution = scipy.optimize.least_squares(equations, x0, bounds = bnds, args=args)
    parameters = {"min": solution.x[0], "max": solution.x[1], "mode": solution.x[2]}
    print(parameters)
    print("Solve equations time: ", time.time() - ti)