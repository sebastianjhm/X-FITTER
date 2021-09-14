import math

class RECIPROCAL:
    """
    Reciprocal distribution
    https://en.wikipedia.org/wiki/Reciprocal_distribution
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
        
    def cdf(self, x):
        """
        Cumulative distribution function.
        Calculated with quadrature integration method of scipy.
        """
        return (math.log(x) - math.log(self.a))/(math.log(self.b) -  math.log(self.a))
    
    def pdf(self, x):
        """
        Probability density function
        """
        return 1/(x*(math.log(self.b) -  math.log(self.a)))
    
    def get_num_parameters(self):
        """
        Number of parameters of the distribution
        """
        return len(self.parameters.keys())
    
    def parameter_restrictions(self):
        """
        Check parameters restrictions
        """
        v1 = self.b > self.a
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
            {"min": *, "max": *}
        """
        
        a = measurements.min - 1e-8
        b = measurements.max + 1e-8
        
        
        parameters = {"a": a , "b": b}
        
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
    path = "../data/data_reciprocal.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = RECIPROCAL(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))