import scipy.optimize
import numpy as np
import scipy.special

class DAGUM_4P:
    """
    Dagum distribution
    https://en.wikipedia.org/wiki/Dagum_distribution
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
        self.p = self.parameters["p"]
        self.loc = self.parameters["loc"]
        
    def cdf(self, x):
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        return (1 + ((x-self.loc)/self.b) ** (-self.a)) ** (-self.p)
    
    def pdf(self, x):
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        return (self.a * self.p / x) * ((((x-self.loc)/self.b) ** (self.a*self.p))/(((((x-self.loc)/self.b) ** (self.a))+1)**(self.p+1)))
        
    def get_num_parameters(self):
        """
        Number of parameters of the distribution
        """
        return len(self.parameters.keys())
    
    def parameter_restrictions(self):
        """
        Check parameters restrictions
        """
        v1 = self.p > 0
        v2 = self.a > 0
        v3 = self.b > 0
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
            {"a": * , "b": *, "c": *}
        """
        frequencies, bin_edges = np.histogram(measurements.data, density=True)

        def sse(parameters):
            def __pdf(x, params): return (params["a"] * params["p"] / (x-params["loc"])) * ((((x-params["loc"])/params["b"]) ** (
                params["a"]*params["p"]))/((((x/params["b"]) ** (params["a"]))+1)**(params["p"]+1)))

            central_values = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]

            ## Calculate fitted PDF and error with fit in distribution
            pdf_values = [__pdf(c, parameters) for c in central_values]

            ## Calculate SSE (sum of squared estimate of errors)
            sse = np.sum(np.power(frequencies - pdf_values, 2))

            return sse
        
        def equations(sol_i, measurements):
            ## Variables declaration
            a, b, p, loc = sol_i
            
            ## Generatred moments function (not-centered)
            miu = lambda k: (b**k) * p * scipy.special.beta((a*p+k)/a, (a-k)/a)
            
            ## Parametric expected expressions
            parametric_mean = miu(1) + loc
            parametric_variance = -(miu(1)**2) + miu(2)
            parametric_skewness = (2*miu(1)**3 - 3*miu(1)*miu(2) + miu(3)) / (-(miu(1)**2) + miu(2))**1.5
            # parametric_kurtosis = (-3*miu(1)**4 + 6*miu(1)**2 * miu(2) -4 * miu(1) * miu(3) + miu(4)) / (-(miu(1)**2) + miu(2))**2
            parametric_median = b * ((2**(1/p))-1) ** (-1/a) + loc
            parametric_mode = b * ((a*p-1)/(a+1)) ** (1/a) + loc
            
            ## System Equations
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            eq3 = parametric_median - measurements.median
            eq4 = parametric_mode - measurements.mode
            
            return (eq1, eq2, eq3, eq4)
        
        ## Scipy Burr3 = Dagum parameter
        s0_burr3_sc = scipy.stats.burr.fit(measurements.data)
        parameters_sc = {"a": s0_burr3_sc[0], "b": s0_burr3_sc[3], "p": s0_burr3_sc[1], "loc": s0_burr3_sc[2]}

        if s0_burr3_sc[0] <= 2:
            return(parameters_sc)
        else:
            a0 = s0_burr3_sc[0]
            x0 = [a0, 1, 1, measurements.mean]
            b = ((1e-5, 1e-5, 1e-5, -np.inf), (np.inf, np.inf, np.inf, np.inf))
            solution = scipy.optimize.least_squares(equations, x0, bounds = b, args=([measurements]))
            parameters_ls = {"a": solution.x[0], "b": solution.x[1], "p": solution.x[2], "loc": solution.x[3]}
            
            sse_sc = sse(parameters_sc)
            sse_ls = sse(parameters_ls)

            if sse_sc < sse_ls:
                return(parameters_sc)
            else:
                return(parameters_ls)

    
if __name__ == '__main__':
    ## Import function to get measurements
    from measurements_cont.measurements import MEASUREMENTS

    ## Import function to get measurements
    def get_data(direction):
        file  = open(direction,'r')
        data = [float(x.replace(",",".")) for x in file.read().splitlines()]
        return data
    
    ## Distribution class
    path = "../data/data_dagum_4p.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = DAGUM_4P(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    
    # a, b, p, loc = [32, 17, 7, 50]
    
    # ## Moments Burr Distribution
    # miu = lambda k: (b**k) * p * scipy.special.beta((a*p+k)/a, (a-k)/a)
    
    # ## Parametric expected expressions
    # parametric_mean = miu(1) + loc
    # parametric_variance = -(miu(1)**2) + miu(2)
    # parametric_skewness = (2*miu(1)**3 - 3*miu(1)*miu(2) + miu(3)) / (parametric_variance)**1.5
    # parametric_kurtosis = (-3*miu(1)**4 + 6*miu(1)**2 * miu(2) -4 * miu(1) * miu(3) + miu(4)) / (parametric_variance)**2
    # parametric_median = b * ((2**(1/p))-1) ** (-1/a) + loc
    # parametric_mode = b * ((a*p-1)/(a+1)) ** (1/a) + loc
    
    # print(parametric_mean, parametric_variance, parametric_skewness, parametric_kurtosis, parametric_median, parametric_mode)
    # print(measurements.mean, measurements.variance, measurements.median, measurements.mode)
    
    
    import scipy.stats
    print(scipy.stats.burr.fit(data))
