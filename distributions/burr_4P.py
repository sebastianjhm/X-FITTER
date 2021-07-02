import scipy.integrate
from scipy.optimize import minimize, least_squares
from scipy.special import beta
import numpy as np
import scipy.stats
import warnings

warnings.filterwarnings("ignore")

class BURR_4P:
    """
    Burr distribution
    Conpendium.pdf pg.27
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.A = self.parameters["A"]
        self.B = self.parameters["B"]
        self.C = self.parameters["C"]
        self.loc = self.parameters["loc"]
        
    def cdf(self, x):
        """
        Cumulative distribution function.
        Calculated with quadrature integration method of scipy.
        """
        return 1 - ((1 + ((x-self.loc)/self.A) ** (self.B )) ** (-self.C))
      
    def pdf(self, x):
        """
        Probability density function
        """
        return ((self.B * self.C)/self.A) * (((x-self.loc)/self.A) ** (self.B - 1)) * ((1 + ((x-self.loc)/self.A) ** (self.B )) ** (-self.C - 1))
        
    def get_num_parameters(self):
        """
        Number of parameters of the distribution
        """
        return len(self.parameters.keys())
    
    def parameter_restrictions(self):
        """
        Check parameters restrictions
        """
        v1 = self.A > 0
        v2 = self.B > 0
        v3 = self.C > 0
        return v1 and v2 and v3

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
            {"A": * , "B": *, "C": *}
        """
        
        # def equations(sol_i, measurements):
        #     ## Variables declaration
        #     A, B, C, loc = sol_i
            
        #     ## Moments Burr Distribution
        #     miu = lambda r: (A**r) * C * beta((B*C-r)/B, (B+r)/B)
            
        #     ## Parametric expected expressions
        #     parametric_mean = miu(1) + loc
        #     parametric_variance = -(miu(1)**2) + miu(2)
        #     # parametric_skewness = 2*miu(1)**3 - 3*miu(1)*miu(2) + miu(3)
        #     parametric_kurtosis = -3*miu(1)**4 + 6*miu(1)**2 * miu(2) -4 * miu(1) * miu(3) + miu(4)
        #     parametric_median = A * ((2**(1/C))-1)**(1/B) + loc
        #     parametric_mode = A*((B - 1)/(B*C + 1))**(1/B) + loc
            
        #     ## System Equations
        #     eq1 = parametric_mean - measurements.mean
        #     eq2 = parametric_variance - measurements.variance
        #     eq3 = parametric_median - measurements.median
        #     eq4 = parametric_kurtosis - measurements.kurtosis
            
        #     return (eq1, eq2, eq3, eq4)
        
        # x0 = [10, 6, 5, measurements.min]
        # b = ((1e-5, 1e-5, 1e-5, -np.inf), (np.inf, np.inf, np.inf, np.inf))
        # solution = least_squares(equations, x0, bounds = b, args=([measurements]))
        # parameters = {"A": solution.x[0], "B": solution.x[1], "C": solution.x[2], "loc": solution.x[3]}
        # print(parameters)
        
        scipy_params = scipy.stats.burr12.fit(measurements.data)
        parameters = {"A": scipy_params[3], "B": scipy_params[0], "C": scipy_params[1], "loc": scipy_params[2]}
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
    path = "..\\data\\data_burr_4P.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = BURR_4P(measurements)
    print(distribution.get_parameters(measurements))

    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))