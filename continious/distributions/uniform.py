class UNIFORM:
    """
    Uniform distribution
    https://en.wikipedia.org/wiki/Continuous_uniform_distribution
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.min_ = self.parameters["min"]
        self.max_ = self.parameters["max"]
        
    def cdf(self, x):
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        return (x - self.min_)/(self.max_ - self.min_)
    
    def pdf(self, x):
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        return 1/(self.max_ - self.min_)
    
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
            {"min": *, "max": *}
        """
        
        _min = measurements.min - 1e-8
        _max = measurements.max + 1e-8
        
        
        parameters = {"min": _min , "max": _max}
        
        return parameters
    
if __name__ == '__main__':
    ## Import function to get measurements
    from measurements_cont.measurements import MEASUREMENTS

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
    print(distribution.pdf(measurements.mean))