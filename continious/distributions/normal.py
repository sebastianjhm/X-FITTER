import scipy.integrate
import math

class NORMAL:
    """
    Normal distribution
    https://en.wikipedia.org/wiki/Normal_distribution          
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
        # result, error = scipy.integrate.quad(self.pdf, float("-inf"), x)
        result = scipy.stats.norm.cdf((x-self.miu)/self.sigma)
        return result
    
    def pdf(self, x):
        """
        Probability density function
        """
        return (1/(self.sigma * math.sqrt(2 * math.pi))) * math.e ** (-(((x - self.miu)**2) / (2*self.sigma**2)))
    
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
            {"miu": *, "variance": *, "skewness": *, "kurtosis": *, "data": *}

        Returns
        -------
        parameters : dict
            {"miu": *, "sigma": *}
        """
        
        μ = measurements.mean
        σ = math.sqrt(measurements.variance)
        
        parameters = {"miu": μ, "sigma": σ}
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
    path = "../data/data_normal.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = NORMAL(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))