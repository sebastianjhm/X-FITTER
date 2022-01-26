from scipy.optimize import least_squares
import math
import scipy.stats
import numpy as np
from scipy.special import beta

class PEARSON_TYPE_6_4P:
    """
    PEARSON TYPE 6 distribution
    pearson_type_6(α1, α2, 1) = prime_beta(α1, α2)
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betaprime.html    
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha1 = self.parameters["alpha1"]
        self.alpha2 = self.parameters["alpha2"]
        self.beta = self.parameters["beta"]
        self.loc = self.parameters["loc"]
        
    def cdf(self, x):
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        # result = scipy.stats.betaprime.cdf(x, self.alpha1, self.alpha2, loc=self.loc, scale=self.beta)
        result = scipy.stats.beta.cdf((x-self.loc)/((x-self.loc)+self.beta),self.alpha1, self.alpha2)
        return result
    
    def pdf(self, x):
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        # result = scipy.stats.betaprime.pdf(x, self.alpha1, self.alpha2, loc=self.loc, scale=self.beta)
        result = (((x-self.loc)/self.beta)**(self.alpha1-1))/(self.beta * beta(self.alpha1,self.alpha2) * (1+(x-self.loc)/self.beta)**(self.alpha1+self.alpha2))
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
        v1 = self.alpha1 > 0
        v2 = self.alpha2 > 0
        v3 = self.beta > 0
        return v1 and v2 and v3
    
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
        ## In this distribution solve the system is best than scipy estimation.
        def equations(sol_i, measurements):
            α1, α2, β, loc = sol_i
        
            parametric_mean = β*α1/(α2-1) + loc
            parametric_variance = (β**2)*α1*(α1+α2-1)/((α2-1)**2 * (α2-2))
            # parametric_skewness = 2*math.sqrt(((α2-2))/(α1*(α1+α2-1)))*(((2*α1+α2-1))/(α2-3))
            parametric_median = loc + β*scipy.stats.beta.ppf(0.5, α1, α2)/(1-scipy.stats.beta.ppf(0.5, α1, α2))
            parametric_mode = β*(α1-1)/(α2+1) + loc
            
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            # eq3 = parametric_skewness - measurements.skewness
            eq3 = parametric_median - measurements.median
            eq4 = parametric_mode - measurements.mode
            
            return (eq1, eq2, eq3, eq4)
        
        try:
            bnds = ((0, 0, 0, -np.inf), (np.inf, np.inf, np.inf, np.inf))
            x0 = (measurements.mean, measurements.mean, measurements.mean, measurements.mean)
            args = ([measurements])
            solution = least_squares(equations, x0, bounds = bnds, args=args)
            parameters = {"alpha1": solution.x[0], "alpha2": solution.x[1], "beta": solution.x[2], "loc": solution.x[3]}
            
            if math.isnan(((measurements.mean/parameters["beta"])**(parameters["alpha1"]-1))/(parameters["beta"] * beta(parameters["alpha1"],parameters["alpha2"]) * (1+measurements.mean/parameters["beta"])**(parameters["alpha1"]+parameters["alpha2"]))):
                scipy_params = scipy.stats.betaprime.fit(measurements.data)
                parameters = {"alpha1": scipy_params[0], "alpha2": scipy_params[1], "beta": scipy_params[3], "loc": scipy_params[2]}
        except ValueError:
            scipy_params = scipy.stats.betaprime.fit(measurements.data)
            parameters = {"alpha1": scipy_params[0], "alpha2": scipy_params[1], "beta": scipy_params[3], "loc": scipy_params[2]}
        return parameters

if __name__ == '__main__':
    ## Import function to get measurements
    from measurements.measurements import MEASUREMENTS

    ## Import function to get measurements
    def get_data(direction):
        file  = open(direction,'r')
        data = [float(x.replace(",",".")) for x in file.read().splitlines()]
        return data
    
    ## Distribution class
    path = "../data/data_pearson_type_6_4p.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = PEARSON_TYPE_6_4P(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
    
    
    
    
    
    
    
    
    
    
    print("\n========= Time parameter estimation analisys ========")
    
    import time
    
    def equations(sol_i, measurements):
        α1, α2, β, loc = sol_i
    
        parametric_mean = β*α1/(α2-1) + loc
        parametric_variance = (β**2)*α1*(α1+α2-1)/((α2-1)**2 * (α2-2))
        # parametric_skewness = 2*math.sqrt(((α2-2))/(α1*(α1+α2-1)))*(((2*α1+α2-1))/(α2-3))
        parametric_median = loc + β*scipy.stats.beta.ppf(0.5, α1, α2)/(1-scipy.stats.beta.ppf(0.5, α1, α2))
        parametric_mode = β*(α1-1)/(α2+1) + loc
        
        eq1 = parametric_mean - measurements.mean
        eq2 = parametric_variance - measurements.variance
        # eq3 = parametric_skewness - measurements.skewness
        eq3 = parametric_median - measurements.median
        eq4 = parametric_mode - measurements.mode
        
        return (eq1, eq2, eq3, eq4)
    
    ti = time.time()
    bnds = ((0, 0, 0, -np.inf), (np.inf, np.inf, np.inf, np.inf))
    x0 = (measurements.mean, measurements.mean, measurements.mean, measurements.mean)
    args = ([measurements])
    solution = least_squares(equations, x0, bounds = bnds, args=args)
    parameters = {"alpha1": solution.x[0], "alpha2": solution.x[1], "beta": solution.x[2], "loc": solution.x[3]}
    print(parameters)
    print("Solve equations time: ", time.time() -ti)
    
    ti = time.time()
    scipy_params = scipy.stats.betaprime.fit(data)
    print(scipy_params)
    parameters = {"alpha1": scipy_params[0], "alpha2": scipy_params[1], "beta": scipy_params[3], "loc": scipy_params[2]}
    print(parameters)
    print("Scipy time get parameters: ",time.time() -ti)
