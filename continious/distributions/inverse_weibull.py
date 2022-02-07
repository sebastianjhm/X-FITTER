import scipy.stats
import math

class INVERSE_WEIBULL:
    """
    Inverse Weibull Distribution
    https://scipy.github.io/devdocs/tutorial/stats/continuous_invweibull.html
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.c = self.parameters["c"]
        self.miu = self.parameters["miu"]
        self.sigma = self.parameters["sigma"]

    def cdf(self, x):
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        z = lambda t: (t - self.miu) / self.sigma
        # result = scipy.stats.invweibull.cdf(x, self.c, loc = self.miu, scale = self.sigma)
        result = math.exp(-z(x) ** -self.c)
        return result
    
    def pdf(self, x):
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        z = lambda t: (t - self.miu) / self.sigma
        # result = scipy.stats.invweibull.pdf(x, self.c, loc = self.miu, scale = self.sigma)
        result = (1/self.sigma) * self.c * z(x) ** (-self.c-1) * math.exp(-z(x) ** -self.c)
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
            {"miu": *, "variance": *, "skewness": *, "kurtosis": *, "data": *}

        Returns
        -------
        parameters : dict
            {"miu": *, "sigma": *}
        """
        scipy_params = scipy.stats.invweibull.fit(measurements.data)
        
        parameters = {"c": scipy_params[0],"miu": scipy_params[1], "sigma": scipy_params[2]}
        return parameters
    
if __name__ == '__main__':
    ## Import function to get measurements
    from measurements_cont.measurements import MEASUREMENTS

    ## Import function to get measurements
    def get_data(direction):
        file  = open(direction,'r')
        data = [float(x.replace(",",".")) for x in file.read().splitlines()]
        return data
    
    ## Distribution class
    path = "../data/data_inverse_weibull.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = INVERSE_WEIBULL(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))