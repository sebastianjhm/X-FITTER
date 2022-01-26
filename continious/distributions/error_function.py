import scipy.stats
import math

class ERROR_FUNCTION:
    """
    Error Function distribution
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.h = self.parameters["h"]
        
    def cdf(self, x):
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        return scipy.stats.norm.cdf((2**0.5) * self.h * x)
    
    def pdf(self, x):
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        return self.h * math.exp(-(self.h*x)**2) / math.sqrt(math.pi)
    
    def get_num_parameters(self):
        """
        Number of parameters of the distribution
        """
        return len(self.parameters.keys())
    
    def parameter_restrictions(self):
        """
        Check parameters restrictions
        """
        v1 = self.h > 0
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
            {"h": *}
        """
        h = math.sqrt(1/(2*measurements.variance))
       
        ## Results
        parameters = {"h": h}

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
    path = "../data/data_error_function.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = ERROR_FUNCTION(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))