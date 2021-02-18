import scipy.stats
import math

class ERROR_FUNCTION:
    """
    Error Function distribution
    https://www.vosesoftware.com/riskwiki/Erfdistribution.php
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.h = self.parameters["h"]
        
    def cdf(self, x):
        """
        Cumulative distribution function.
        Calculated with quadrature integration method of scipy.
        """
        return scipy.stats.norm.cdf((2**0.5) * self.h * x)
    
    def pdf(self, x):
        """
        Probability density function
        """
        return self.h * math.exp(-(self.h*x)**2) / math.sqrt(math.pi)
    
    def get_num_parameters(self):
        """
        Number of parameters of the distribution
        """
        return len(self.parameters.keys())
    
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
            {"h": *}
        """
        h = math.sqrt(1/(2*measurements["variance"]))
       
        ## Results
        parameters = {"h": h}

        return parameters
    
# def getData(direction):
#     file  = open(direction,'r')
#     data = [float(x.replace(",",".")) for x in file.read().splitlines()]
#     return data

# path = "C:\\Users\\USUARIO1\\Desktop\\Fitter\\data\\data_error_function.txt"
# data = getData(path) 
# measurements = get_measurements(data)
# distribution = ERROR_FUNCTION(measurements)
# print(distribution.get_parameters(measurements))
# print(distribution.cdf(-0.12))