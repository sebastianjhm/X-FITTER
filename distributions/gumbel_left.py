import math
from scipy.optimize import fsolve

class GUMBEL_LEFT:
    """
    Gumbel Left Distribution
    Gumbel Min Distribution
    Extreme Value Minimum Distribution
    https://mathworld.wolfram.com/GumbelDistribution.html
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        
        self.miu = self.parameters["miu"]
        self.sigma = self.parameters["sigma"]
        
    def cdf(self, x):
        """
        Cumulative distribution function.
        Calculated with quadrature integration method of scipy.
        """
        z = lambda x: (x - self.miu) / self.sigma
        return 1 - math.exp(-math.exp(z(x)))
    
    def pdf(self, x):
        """
        Probability density function
        """
        z = lambda x: (x - self.miu) / self.sigma
        return (1/self.sigma) * math.exp(z(x)+math.exp(-z(x)))
    
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
            {"mean": *, "variance": *, "skewness": *, "kurtosis": *, "data": *}

        Returns
        -------
        parameters : dict
            {"c": *, "miu": *, "sigma": *}
        """
        def equations(sol_i, measurements):
            ## Variables declaration
            miu, sigma = sol_i
            
            ## Parametric expected expressions
            parametric_mean = miu - sigma * 0.5772156649
            parametric_variance = (sigma ** 2) * (math.pi ** 2)/6
            
            ## System Equations
            eq1 = parametric_mean - measurements["mean"]
            eq2 = parametric_variance - measurements["variance"]
            
            return (eq1, eq2)
        
        solution =  fsolve(equations, (1, 1), measurements)
        parameters = {"miu": solution[0], "sigma": solution[1]}
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
    path = "..\\data\\data_gumbel_left.txt"
    data = get_data(path) 
    measurements = get_measurements(data)
    distribution = GUMBEL_LEFT(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(70))