from scipy.optimize import least_squares
import math
import scipy.stats
import numpy as np
from scipy.special import beta

class PEARSON_TYPE_6:
    """
    PEARSON TYPE 6 distribution
    https://www.vosesoftware.com/riskwiki/PearsonType6distribution.php    
    pearson_type_6(α1, α2, 1) = prime_beta(α1, α2)
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha1 = self.parameters["alpha1"]
        self.alpha2 = self.parameters["alpha2"]
        self.beta = self.parameters["beta"]
        
    def cdf(self, x):
        """
        Cumulative distribution function
        Calculated with quadrature integration method of scipy
        """
        # result = scipy.stats.betaprime.cdf(x, self.alpha1, self.alpha2, scale=self.beta)
        result = scipy.stats.beta.cdf(x/(x+self.beta),self.alpha1, self.alpha2)
        return result
    
    def pdf(self, x):
        """
        Probability density function
        """
        # result = scipy.stats.betaprime.pdf(x, self.alpha1, self.alpha2, scale=self.beta)
        result = ((x/self.beta)**(self.alpha1-1))/(self.beta * beta(self.alpha1,self.alpha2) * (1+x/self.beta)**(self.alpha1+self.alpha2))
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
            {"mean": *, "variance": *, "skewness": *, "kurtosis": *, "data": *}

        Returns
        -------
        parameters : dict
            {"alpha": *, "beta": *, "min": *, "max": *}
        """
        def equations(sol_i, data_mean, data_variance, data_skewness):
            α1, α2, β = sol_i
        
            parametric_mean = β*α1/(α2-1)
            parametric_variance = (β**2)*α1*(α1+α2-1)/((α2-1)**2 * (α2-2))
            parametric_skewness = 2*math.sqrt(((α2-2))/(α1*(α1+α2-1)))*(((2*α1+α2-1))/(α2-3))
        
            eq1 = parametric_mean - data_mean
            eq2 = parametric_variance - data_variance
            eq3 = parametric_skewness - data_skewness
        
            return (eq1, eq2, eq3)
        
        try:
            bnds = ((0, 0, 0), (np.inf, np.inf, np.inf))
            x0 = (10*measurements["mean"], 10*measurements["mean"], measurements["mean"])
            args = (measurements["mean"], measurements["variance"], measurements["skewness"])
            solution = least_squares(equations, x0, bounds = bnds, args=args)
            parameters = {"alpha1": solution.x[0], "alpha2": solution.x[1], "beta": solution.x[2]}
            
            if math.isnan(((measurements["mean"]/parameters["beta"])**(parameters["alpha1"]-1))/(parameters["beta"] * beta(parameters["alpha1"],parameters["alpha2"]) * (1+measurements["mean"]/parameters["beta"])**(parameters["alpha1"]+parameters["alpha2"]))):
                scipy_params = scipy.stats.betaprime.fit(measurements["data"])
                parameters = {"alpha1": scipy_params[0], "alpha2": scipy_params[1], "beta": scipy_params[3]}
        except ValueError:
            scipy_params = scipy.stats.betaprime.fit(measurements["data"])
            parameters = {"alpha1": scipy_params[0], "alpha2": scipy_params[1], "beta": scipy_params[3]}
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
    path = "..\\data\\data_pearson_type_6.txt"
    data = get_data(path) 
    measurements = get_measurements(data)
    distribution = PEARSON_TYPE_6(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements["mean"]))
    print(distribution.pdf(measurements["mean"]))
    
    print("========= Time parameter estimation analisys ========")
    
    import time
    
    def equations(sol_i, data_mean, data_variance, data_skewness):
        α1, α2, β = sol_i
    
        parametric_mean = β*α1/(α2-1)
        parametric_variance = (β**2)*α1*(α1+α2-1)/((α2-1)**2 * (α2-2))
        parametric_skewness = 2*math.sqrt(((α2-2))/(α1*(α1+α2-1)))*(((2*α1+α2-1))/(α2-3))
    
        eq1 = parametric_mean - data_mean
        eq2 = parametric_variance - data_variance
        eq3 = parametric_skewness - data_skewness
    
        return (eq1, eq2, eq3)
    
    ti = time.time()
    bnds = ((0, 0, 0), (np.inf, np.inf, np.inf))
    x0 = (10*measurements["mean"], 10*measurements["mean"], measurements["mean"])
    args = (measurements["mean"], measurements["variance"], measurements["skewness"])
    solution = least_squares(equations, x0, bounds = bnds, args=args)
    parameters = {"alpha1": solution.x[0], "alpha2": solution.x[1], "beta": solution.x[2]}
    print(parameters)
    print("Solve equations time: ", time.time() -ti)
    
    ti = time.time()
    scipy_params = scipy.stats.betaprime.fit(data)
    parameters = {"alpha1": scipy_params[0], "alpha2": scipy_params[1], "beta": scipy_params[2]}
    print(parameters)
    print("Scipy time get parameters: ",time.time() -ti)