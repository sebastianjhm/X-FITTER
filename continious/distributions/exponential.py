import math

class EXPONENTIAL:
    """
    Exponential distribution
    https://en.wikipedia.org/wiki/Exponential_distribution         
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.lambda_ = self.parameters["lambda"]
        
    def cdf(self, x):
        """
        Cumulative distribution function.
        Calculated with known formula.
        """
        return 1 - math.e ** (-self.lambda_ * x)
    
    def pdf(self, x):
        """
        Probability density function
        """
        return self.lambda_ * math.e ** (-self.lambda_ * x)
    
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
        The parameters are calculated by solving the equations of the measures expected 
        for this distribution.The number of equations to consider is equal to the number 
        of parameters.
        
        Parameters
        ----------
        measurements : dict
            {"mean": *, "variance": *, "skewness": *, "kurtosis": *, "data": *}

        Returns
        -------
        parameters : dict
            {"lambda": *}
        """
        λ = 1/measurements.mean
        parameters = {"lambda": λ}
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
    path = "../data/data_exponential.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = EXPONENTIAL(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
    