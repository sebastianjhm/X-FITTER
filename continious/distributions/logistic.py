import scipy.special as sc
import scipy.stats
import math

class LOGISTIC:
    """
    Logistic distribution
    https://en.wikipedia.org/wiki/Logistic_distribution
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
        z = lambda x: math.exp(-(x-self.miu)/self.sigma)
        result = 1/(1+z(x))
        return result
    
    def pdf(self, x):
        """
        Probability density function
        """
        z = lambda x: math.exp(-(x-self.miu)/self.sigma)
        result = z(x)/(self.sigma*(1+z(x))**2)
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
            {"miu": *, "sigma": *}
        """
        μ = measurements.mean
        σ = math.sqrt(3*measurements.variance/(math.pi**2))
        
        ## Results
        parameters = {"miu": μ, "sigma": σ}

        return parameters

if __name__ == "__main__":   
    ## Import function to get measurements
    from measurements.measurements import MEASUREMENTS
    
    ## Import function to get measurements
    def get_data(direction):
        file  = open(direction,'r')
        data = [float(x.replace(",",".")) for x in file.read().splitlines()]
        return data
    
    ## Distribution class
    path = "..\\data\\data_logistic.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = LOGISTIC(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))