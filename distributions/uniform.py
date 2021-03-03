class UNIFORM:
    """
    Uniform distribution
    https://www.vosesoftware.com/riskwiki/Gammadistribution.php
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
        return (x - self.min_)/(self.max_ - self.min_)
    
    def pdf(self, x):
        """
        Probability density function
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
            {"mean": *, "variance": *, "skewness": *, "kurtosis": *, "data": *}

        Returns
        -------
        parameters : dict
            {"min": *, "max": *}
        """
        
        _min = min(measurements["data"]) - 1e-8
        _max = max(measurements["data"]) + 1e-8
        
        
        parameters = {"min": _min , "max": _max}
        
        return parameters