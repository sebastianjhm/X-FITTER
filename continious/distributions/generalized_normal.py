import scipy.stats
from scipy.special import gammaincc
from scipy.special import gamma

class GENERALIZED_NORMAL:
    """
    Generalized normal distribution
    https://en.wikipedia.org/wiki/Generalized_normal_distribution
    https://www.vosesoftware.com/riskwiki/Errordistribution.php
    This distribution is known whit the following names:
    * Error Distribution
    * Exponential Power Distribution
    * Generalized Error Distribution (GED)
    * Generalized Gaussian distribution (GGD) 
    * Subbotin distribution
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        
        self.beta = self.parameters["beta"]
        self.miu = self.parameters["miu"]
        self.alpha = self.parameters["alpha"]
        
    def cdf(self, x):
        """
        Cumulative distribution function.
        Calculated with quadrature integration method of scipy.
        """
        return scipy.stats.gennorm.cdf(x , self.beta, loc=self.miu, scale=self.alpha)
    
    def pdf(self, x):
        """
        Probability density function
        """
        return scipy.stats.gennorm.pdf(x , self.beta, loc=self.miu, scale=self.alpha)
    
    def get_num_parameters(self):
        """
        Number of parameters of the distribution
        """
        return len(self.parameters.keys())

    def parameter_restrictions(self):
        """
        Check parameters restriction
        """
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2
    
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
            {"beta": *, "miu": *, "alpha": *}
        """
        scipy_params = scipy.stats.gennorm.fit(measurements.data)
        parameters = {"beta": scipy_params[0], "miu": scipy_params[1], "alpha": scipy_params[2]}
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
    path = "../data/data_generalized_normal.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = GENERALIZED_NORMAL(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
