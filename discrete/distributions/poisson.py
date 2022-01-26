import scipy.stats
import math

class POISSON:
    """
    Poisson distribution
    https://en.wikipedia.org/wiki/Poisson_distribution
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.lambda_ = self.parameters["lambda"]
                
    def cdf(self, x):
        """
        Probability density function
        Calculated using the definition of the function
        Alternative: scipy cdf method
        """
        result = scipy.stats.poisson.cdf(x, self.lambda_)
        return result

    
    def pdf(self, x):
        """
        Probability density function
        Calculated using the definition of the function
        """
        result = (self.lambda_**x)*math.exp(-self.lambda_)/math.factorial(x)
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
        v1 = self.lambda_ > 0
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
            {"alpha": *, "beta": *, "gamma": *}
        """
        λ = measurements.mean
        parameters = {"lambda": λ}
        return parameters

if __name__ == '__main__':
    ## Import function to get measurements
    from measurements.measurements import MEASUREMENTS

    ## Import function to get measurements
    def get_data(direction):
        file  = open(direction,'r')
        data = [int(x) for x in file.read().splitlines()]
        return data

    ## Distribution class
    path = "../data/data_poisson.txt"
    data = get_data(path)
    measurements = MEASUREMENTS(data)
    distribution = POISSON(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(round(measurements.mean)))
    print(distribution.pdf(round(measurements.mean)))