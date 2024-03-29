import scipy.integrate
import math
import scipy.stats
import scipy.special as sc

class CHI_SQUARE:
    """
    Chi Square distribution
    https://en.wikipedia.org/wiki/Chi-square_distribution          
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.df = self.parameters["df"]
        
    def cdf(self, x):
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        # result, error = scipy.integrate.quad(self.pdf, 0, x)
        # result = scipy.stats.chi2.cdf(x, self.df)
        result = sc.gammainc(self.df/2, x/2)
        return result
    
    def pdf(self, x):
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        # result = (1/(2**(self.df/2) * math.gamma(self.df/2))) * (x**((self.df/2)-1)) * (math.e ** (-x/2))
        # print(result)
        result = scipy.stats.chi2.pdf(x, self.df)
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
        v1 = self.df > 0
        v2 = type(self.df) == int
        return v1 and v2

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
            {"df": *}
        """
        parameters = {"df": round(measurements.mean)}
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
    path = "../data/data_chi_square.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = CHI_SQUARE(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
    
    print(scipy.stats.chi2.fit(measurements.data))