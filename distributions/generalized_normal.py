import scipy.stats


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
        scipy_params = scipy.stats.gennorm.fit(measurements["data"])
        parameters = {"beta": scipy_params[0], "miu": scipy_params[1], "alpha": scipy_params[2]}
        return parameters
    
print(scipy.stats.gamma.ppf(0.48, 24, scale=2))

def get_measurements(data: list) -> dict:
    import scipy.stats
    import numpy as np
    measurements = {}
    
    miu_3 = scipy.stats.moment(data, 3)
    miu_4 = scipy.stats.moment(data, 4)
    mean = np.mean(data)
    variance = np.var(data, ddof=1)
    skewness = miu_3 / pow(np.std(data, ddof=1),3)
    kurtosis = miu_4 / pow(np.std(data, ddof=1),4)
    median = np.median(data)
    mode = scipy.stats.mode(data)[0][0]
    
    measurements["mean"] = mean
    measurements["variance"] =  variance
    measurements["skewness"] = skewness
    measurements["kurtosis"] = kurtosis
    measurements["data"] = data
    measurements["median"] = median
    measurements["mode"] = mode
    
    return measurements

def getData(direction):
    file  = open(direction,'r')
    data = [float(x.replace(",",".")) for x in file.read().splitlines()]
    return data

path = "C:\\Users\\USUARIO1\\Desktop\\Fitter\\data\\data_generalized_normal.txt"
data = getData(path) 

measurements = get_measurements(data)
distribution = GENERALIZED_NORMAL(measurements)
print(distribution.get_parameters(measurements))
print(distribution.cdf(5.4455))
