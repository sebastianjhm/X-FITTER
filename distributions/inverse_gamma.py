import scipy.stats
import math
import scipy.special as sc

class INVERSE_GAMMA:
    """
    Inverse Gamma distribution
    Also known Pearson Type 5 distribution
    https://en.wikipedia.org/wiki/Inverse-gamma_distribution    
    https://www.vosesoftware.com/riskwiki/PearsonType5distribution.php
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
        
    def cdf(self, x):
        """
        Cumulative distribution function.
        Calculated with quadrature integration method of scipy.
        """
        ## Method 1: Integrate PDF function
        # result, error = scipy.integrate.quad(self.pdf, 0, x)
        # print(result)
        
        ## Method 2: Scipy Gamma Distribution class
        # result = scipy.stats.invgamma.cdf(x, a=self.alpha, scale=self.beta)
        # print(result)
        
        upper_inc_gamma = lambda a, x: sc.gammaincc(a, x) * math.gamma(a)
        result = upper_inc_gamma(self.alpha, self.beta/x)/math.gamma(self.alpha)
        return result
    
    def pdf(self, x):
        """
        Probability density function
        """
        return ((self.beta ** self.alpha) * (x**(-self.alpha-1)) * math.e ** (-(self.beta/x))) / math.gamma(self.alpha)
    
    def get_num_parameters(self):
        """
        Number of parameters of the distribution
        """
        return len(self.parameters.keys())
    
    def parameter_restrictions(self):
        """
        Check parameters restrictions
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
            {"alpha": *, "beta": *}
        """
        ## Method 1: Solve system equations with theoric mean and variance
        ## Note: bad estimation. All test reject
        # mean = measurements["mean"]
        # variance = measurements["variance"]
        # alpha = mean ** 2 / variance + 2
        # beta = mean ** 3 / variance + mean
        # parameters = {"alpha": alpha, "beta": beta}
        
        scipy_params = scipy.stats.invgamma.fit(measurements["data"])
        parameters = {"alpha": scipy_params[0], "beta": scipy_params[2]}
        return parameters
    



if __name__ == "__main__":
    ## PPF of inverse gamma
    alpha, beta = 5, 9
    probability = 0.5
    print("Scipy method:", scipy.stats.invgamma.ppf(probability, a=alpha, scale=beta))
    print("Inverse by gamma method", 1/scipy.stats.gamma.ppf((1-probability), a=alpha, scale=1/beta))
    
    ## Import function to get measurements
    from measurements.data_measurements import get_measurements
    
    ## Import function to get measurements
    def getData(direction):
        file  = open(direction,'r')
        data = [float(x.replace(",",".")) for x in file.read().splitlines()]
        return data
    
    ## Distribution class
    path = "..\\data\\data_inverse_gamma.txt"
    data = getData(path)
    measurements = get_measurements(data)
    distribution = INVERSE_GAMMA(measurements)
    
    print(distribution.get_parameters(measurements))
    import time
    ti = time.time()
    print(scipy.stats.invgamma.fit(data))
    print("Time estimation scipy", time.time()-ti)
    
    
    print(distribution.cdf(3.339))
    print(distribution.cdf(measurements["mean"]))
    
    
