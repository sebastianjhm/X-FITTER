import scipy.integrate
import math

class ERLANG:
    """
    Gamma distribution
    https://www.vosesoftware.com/riskwiki/Erlangdistribution.php        
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.m = self.parameters["m"]
        self.beta = self.parameters["beta"]
        
    def cdf(self, x):
        """
        Cumulative distribution function.
        Calculated with quadrature integration method of scipy.
        """
        result, error = scipy.integrate.quad(self.pdf, 0, x)
        return result
    
    def pdf(self, x):
        """
        Probability density function
        """
        return ((self.beta ** -self.m) * (x**(self.m-1)) * math.e ** (-(x / self.beta))) / math.factorial(self.m-1)
    
    def get_num_parameters(self):
        """
        Number of parameters of the distribution
        """
        return len(self.parameters.keys())
    
    def parameter_restrictions(self):
        """
        Check parameters restriction
        """
        v1 = self.m > 0
        v2 = self.beta > 0
        v3 = type(self.m) == int
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
            {"m": *, "beta": *}
        """
        mean = measurements["mean"]
        variance = measurements["variance"]
        
        m = round(mean ** 2 / variance)
        beta = variance / mean
        parameters = {"m": m , "beta": beta}

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
    path = "..\\data\\data_erlang.txt"
    data = get_data(path) 
    measurements = get_measurements(data)
    distribution = ERLANG(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements["mean"]))