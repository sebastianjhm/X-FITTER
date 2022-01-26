import math
from scipy.optimize import fsolve
import scipy.special as sc
import scipy.stats

class CHI_SQUARE_3P:
    """
    Chi Square distribution
    https://en.wikipedia.org/wiki/Chi-square_distribution          
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.df = self.parameters["df"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]
        
    def cdf(self, x):
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        # result, error = scipy.integrate.quad(self.pdf, 0, x)
        # result = scipy.stats.chi2.cdf(x, self.df, self.loc, self.scale)
        result = sc.gammainc(self.df/2, (x-self.loc)/(2*self.scale))
        return result
    
    def pdf(self, x):
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        # result = scipy.stats.chi2.pdf(x, self.df, loc=self.loc, scale=self.scale)
        z = lambda x: (x-self.loc)/self.scale
        result = (1/self.scale)*(1/(2**(self.df/2) * math.gamma(self.df/2))) * (z(x)**((self.df/2)-1)) * (math.e ** (-z(x)/2))
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
        v1 = self.df > 0
        v2 = type(self.df) == int
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
            {"df": *}
        """
        # def equations(sol_i, measurements):
        #     ## Variables declaration
        #     df, loc, scale = sol_i
            
        #     ## Parametric expected expressions
        #     parametric_mean = df * scale + loc
        #     parametric_variance = 2 * df * (scale**2)
        #     parametric_skewness = math.sqrt(8/df)
        #     # parametric_kurtosis = 12 / df +3
            
        #     ## System Equations
        #     eq1 = parametric_mean - measurements.mean
        #     eq2 = parametric_variance - measurements.variance
        #     eq3 = parametric_skewness - measurements.skewness
        #     # eq4 = parametric_kurtosis  - measurements.kurtosis
            
        #     return(eq1, eq2, eq3)
        
        # solution =  fsolve(equations, (1, 1, 1), measurements)
        # print(solution)
        
        # ## Method 1: Solve system
        # df = 8/(measurements.skewness ** 2)
        # scale = math.sqrt(measurements.variance/(2*df))
        # loc = measurements.mean - df * scale
        # parameters = {"df": df, "loc": loc, "scale": scale}
        
        ## Scipy FIT
        scipy_params = scipy.stats.chi2.fit(measurements.data)
        parameters = {"df": scipy_params[0], "loc": scipy_params[1], "scale": scipy_params[2]}
        
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
    path = "../data/data_chi_square_3p.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = CHI_SQUARE_3P(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
