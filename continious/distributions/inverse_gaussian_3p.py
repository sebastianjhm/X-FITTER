import math
import scipy.stats

class INVERSE_GAUSSIAN_3P:
    """
    Inverse Gaussian distribution
    Also known like Wald distribution
    https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution        
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.miu = self.parameters["miu"]
        self.lambda_ = self.parameters["lambda"]
        self.loc = self.parameters["loc"]
        
    def cdf(self, x):
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        result = scipy.stats.norm.cdf(math.sqrt(self.lambda_/(x-self.loc))*(((x-self.loc)/self.miu)-1)) + math.exp(2*self.lambda_/self.miu)*scipy.stats.norm.cdf(-math.sqrt(self.lambda_/(x-self.loc))*(((x-self.loc)/self.miu)+1))
        return result
    
    def pdf(self, x):
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        result = math.sqrt(self.lambda_/(2*math.pi*(x-self.loc)**3)) * math.exp(-(self.lambda_*((x-self.loc)-self.miu)**2)/(2*self.miu**2*(x-self.loc)))
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
        v1 = self.miu > 0
        v2 = self.lambda_ > 0
        return v1 and v2

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
            {"miu": *, "lambda": *}
        """
        
        μ = 3 * math.sqrt(measurements.variance/(measurements.skewness**2))
        λ = μ**3/measurements.variance
        loc = measurements.mean - μ
        
        parameters = {"miu": μ, "lambda": λ, "loc": loc}
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
    path = "../data/data_inverse_gaussian_3p.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = INVERSE_GAUSSIAN_3P(measurements)
    
    print(distribution.get_parameters(measurements))
    print(scipy.stats.invgauss.fit(data))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
