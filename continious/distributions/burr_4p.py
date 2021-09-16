from scipy.optimize import least_squares
import numpy as np
import scipy.stats
import scipy.special as sc

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
        def equations(sol_i, measurements):
            ## Variables declaration
            A, B, C, loc = sol_i
            
            ## Moments Burr Distribution
            miu = lambda r: (A**r) * C * sc.beta((B*C-r)/B, (B+r)/B)
            
            ## Parametric expected expressions
            parametric_mean = miu(1) + loc
            parametric_variance = -(miu(1)**2) + miu(2)
            # parametric_skewness = 2*miu(1)**3 - 3*miu(1)*miu(2) + miu(3)
            parametric_kurtosis = -3*miu(1)**4 + 6*miu(1)**2 * miu(2) -4 * miu(1) * miu(3) + miu(4)
            # parametric_median = A * ((2**(1/C))-1)**(1/B) + loc
            parametric_mode = A*((B - 1)/(B*C + 1))**(1/B) + loc
            
            ## System Equations
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            eq3 = parametric_kurtosis - measurements.kurtosis
            eq4 = parametric_mode - measurements.mode
            
            return (eq1, eq2, eq3, eq4)
        
        ## Solve equations system
        # x0 = [measurements.mean, measurements.mean, measurements.mean, measurements.mean]
        # b = ((1e-5, 1e-5, 1e-5, -np.inf), (np.inf, np.inf, np.inf, np.inf))
        # solution = least_squares(equations, x0, bounds = b, args=([measurements]))
        # parameters = {"A": solution.x[0], "B": solution.x[1], "C": solution.x[2], "loc": solution.x[3]}
        # print(parameters)
        
        ## Scipy class
        # scipy_params = scipy.stats.burr12.fit(measurements.data)
        # parameters = {"A": scipy_params[3], "B": scipy_params[0], "C": scipy_params[1], "loc": scipy_params[2]}
        
        ## This is a exact copy of the scipy burr12_gen founded in
        ## https://github.com/scipy/scipy/blob/master/scipy/stats/_continuous_distns.py
        ## The reason of that is beacause the version 0.18.0 of pyodide run scipy 0.17 and in 
        ## this version doesn't exist BURR12
        from scipy.stats import rv_continuous
        class burr12_gen(rv_continuous):
            def _pdf(self, x, c, d):
                # burr12.pdf(x, c, d) = c * d * x**(c-1) * (1+x**(c))**(-d-1)
                return np.exp(self._logpdf(x, c, d))
        
            def _logpdf(self, x, c, d):
                return np.log(c) + np.log(d) + sc.xlogy(c - 1, x) + sc.xlog1py(-d-1, x**c)
        
            def _cdf(self, x, c, d):
                return -sc.expm1(self._logsf(x, c, d))
        
            def _logcdf(self, x, c, d):
                return sc.log1p(-(1 + x**c)**(-d))
        
            def _sf(self, x, c, d):
                return np.exp(self._logsf(x, c, d))
        
            def _logsf(self, x, c, d):
                return sc.xlog1py(-d, x**c)
        
            def _ppf(self, q, c, d):
                # The following is an implementation of
                #   ((1 - q)**(-1.0/d) - 1)**(1.0/c)
                # that does a better job handling small values of q.
                return sc.expm1(-1/d * sc.log1p(-q))**(1/c)
        
            def _munp(self, n, c, d):
                nc = 1. * n / c
                return d * sc.beta(1.0 + nc, d - nc)
        
        burr12 = burr12_gen(a=0.0, name='burr12')
        scipy_params = burr12.fit(measurements.data)
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
    path = "../data/data_burr_4P.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = BURR_4P(measurements)
    print(distribution.get_parameters(measurements))

    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
