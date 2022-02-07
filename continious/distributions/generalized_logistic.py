import scipy.stats
from scipy.special import digamma, polygamma
import math
import scipy.optimize
import numpy as np

class GENERALIZED_LOGISTIC:
    """
    Generalized Logistic Distribution
    Compendium of Common Probability Distributions (pag.41) ... Michael P. McLaughlin  
    https://docs.scipy.org/doc/scipy/reference/tutorial/stats/continuous_genlogistic.html
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        
        self.alpha = self.parameters["alpha"]
        self.beta= self.parameters["beta"]
        self.gamma = self.parameters["gamma"]
        
    def cdf(self, x):
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        # return scipy.stats.genlogistic.cdf(x, self.gamma, loc=self.alpha, scale=self.beta)
        return 1/((1 + math.exp(-(x-self.alpha)/self.beta))**self.gamma)
    
    def pdf(self, x):
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        # return scipy.stats.genlogistic.pdf(x, self.gamma, loc=self.alpha, scale=self.beta)
        return  (self.gamma/self.beta) * math.exp((-x+self.alpha)/self.beta) * ((1 + math.exp((-x+self.alpha)/self.beta))**(-self.gamma-1))
    
    def get_num_parameters(self):
        """
        Number of parameters of the distribution
        """
        return len(self.parameters.keys())
    
    def parameter_restrictions(self):
        """
        Check parameters restrictions
        """
        v1 = self.beta > 0
        v2 = self.gamma > 0
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
            {"alpha": *, "beta": *, "gamma": *}
        """
        def equations(sol_i, measurements):
            ## Variables declaration
            alpha, beta, gamma = sol_i
            
            ## Parametric expected expressions
            parametric_mean = alpha + beta*(0.57721 + digamma(gamma))
            parametric_variance = beta**2 * (math.pi**2/6+polygamma(1, gamma))
            # parametric_skewness = (polygamma(2,1)+polygamma(2,gamma))/((math.pi**2/6+polygamma(1, gamma))**1.5)
            # parametric_kurtosis = 3+(math.pi**4/15+polygamma(3,gamma))/((math.pi**2/6+polygamma(1, gamma))**2)
            parametric_median = alpha + beta*(-math.log(0.5 ** (-1/gamma)-1))
            # parametric_mode = alpha + beta * math.log(gamma)

            ## System Equations
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            # eq3 = parametric_skewness - measurements.skewness
            # eq3 = parametric_kurtosis - measurements.kurtosis
            eq3 = parametric_median - measurements.median
            # eq3 = parametric_mode - measurements.mode
            
            return (eq1, eq2, eq3)

        # ## scipy.optimize.fsolve methods
        # solution =  scipy.optimize.fsolve(equations, (1, 1, 1), measurements)
        # parameters = {"alpha": solution[0], "beta": solution[1], "gamma": solution[2]}
        # print(parameters)
        
        ## least square methods
        x0 = [measurements.mean, measurements.mean, measurements.mean]
        b = ((1e-5, 1e-5, -np.inf), (np.inf, np.inf, np.inf))
        solution = scipy.optimize.least_squares(equations, x0, bounds = b, args=([measurements]))
        parameters = {"alpha": solution.x[0], "beta": solution.x[1], "gamma": solution.x[2]}
        print(parameters)
        
        # ## scipy methods
        # scipy_params = scipy.stats.genlogistic.fit(measurements.data)
        # parameters = {"alpha": scipy_params[1], "beta": scipy_params[2], "gamma": scipy_params[0]}
        
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
    path = "../data/data_generalized_logistic.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = GENERALIZED_LOGISTIC(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))