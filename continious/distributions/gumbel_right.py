import math
import scipy.optimize

class GUMBEL_RIGHT:
    """
    Gumbel Right Distribution
    Gumbel Max Distribution
    Extreme Value Maximum Distribution
    Compendium of Common Probability Distributions (pag.43) ... Michael P. McLaughlin  
    https://en.wikipedia.org/wiki/Gumbel_distribution
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        
        self.miu = self.parameters["miu"]
        self.sigma = self.parameters["sigma"]
        
    def cdf(self, x):
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        z = lambda t: (t - self.miu) / self.sigma
        return math.exp(-math.exp(-z(x)))
    
    def pdf(self, x):
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        z = lambda t: (t - self.miu) / self.sigma
        return (1/self.sigma) * math.exp(-z(x)-math.exp(-z(x)))
    
    def get_num_parameters(self):
        """
        Number of parameters of the distribution
        """
        return len(self.parameters.keys())
    
    def parameter_restrictions(self):
        """
        Check parameters restrictions
        """
        v1 = self.sigma > 0
        return v1

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
            {"c": *, "miu": *, "sigma": *}
        """
        def equations(sol_i, measurements):
            ## Variables declaration
            miu, sigma = sol_i
            
            ## Parametric expected expressions
            parametric_mean = miu + sigma * 0.5772156649
            parametric_variance = (sigma ** 2) * (math.pi ** 2)/6
            
            ## System Equations
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            
            return (eq1, eq2,)
        
        solution =  scipy.optimize.fsolve(equations, (1, 1), measurements)
        parameters = {"miu": solution[0], "sigma": solution[1]}
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
    path = "../data/data_gumbel_right.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = GUMBEL_RIGHT(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))