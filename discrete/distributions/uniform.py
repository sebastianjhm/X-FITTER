class UNIFORM:
    """
    Uniform distribution
    https://en.wikipedia.org/wiki/Discrete_uniform_distribution
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.min_ = self.parameters["min"]
        self.max_ = self.parameters["max"]
        
    def cdf(self, x):
        """
        Cumulative distribution function.
        Calculated with quadrature integration method of scipy.
        """
        return (x - self.min_ + 1)/(self.max_ - self.min_ + 1)
    
    def pdf(self, x):
        """
        Probability density function
        """
        return 1/(self.max_ - self.min_ + 1)
    
    def get_num_parameters(self):
        """
        Number of parameters of the distribution
        """
        return len(self.parameters.keys())
    
    def parameter_restrictions(self):
        """
        Check parameters restrictions
        """
        v1 = self.max_ > self.min_
        v2 = type(self.max_) == int
        v3 = type(self.min_) == int
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
            {"min": *, "max": *}
        """
        
        _min = round(measurements.min)
        _max = round(measurements.max)
        
        
        parameters = {"min": _min , "max": _max}
        
        return parameters
    
if __name__ == '__main__':
    ## Import function to get measurements
    from measurements.measurements import MEASUREMENTS

    ## Import function to get measurements
    def get_data(direction):
        file  = open(direction,'r')
        data = [float(x.replace(",",".")) for x in file.read().splitlines()]
        return data

    path = "../data/data_uniform.txt"

    ## Distribution class
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = UNIFORM(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))