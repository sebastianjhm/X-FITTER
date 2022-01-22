import math
import scipy.stats
import scipy.special as sc

class ARGUS:
    """
    Alpha distribution
    http://bayanbox.ir/view/5343019340232060584/Norman-L.-Johnson-Samuel-Kotz-N.-Balakrishnan-BookFi.org.pdf          
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.chi= self.parameters["chi"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]
        
    def cdf(self, x):
        """
        Cumulative distribution function
        Calculated with quadrature integration method of scipy
        """
        z = lambda x: (x - self.loc)/self.scale
        # Ψ = lambda t: scipy.stats.norm.cdf(t) - t * scipy.stats.norm.pdf(t) - 0.5
        # print(scipy.stats.argus.cdf(x, self.chi, loc=self.loc, scale=self.scale))
        # print(1 - Ψ(self.chi * math.sqrt(1 - z(x) * z(x))) / Ψ(self.chi))
        result = 1 - sc.gammainc(1.5, self.chi*self.chi*(1- z(x) * z(x))/2)/sc.gammainc(1.5, self.chi*self.chi/2)
        return result
    
    def pdf(self, x):
        """
        Probability density function
        """
        z = lambda x: (x - self.loc)/self.scale
        Ψ = lambda t: scipy.stats.norm.cdf(t) - t * scipy.stats.norm.pdf(t) - 0.5
        # print(scipy.stats.argus.pdf(x, self.chi, loc=self.loc, scale=self.scale))
        result = (1/self.scale) * ((self.chi ** 3) / (math.sqrt(2*math.pi) * Ψ(self.chi))) * z(x) * math.sqrt(1-z(x)*z(x)) * math.exp(-0.5 * self.chi ** 2 * (1-z(x)*z(x)))
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
        v1 = self.chi > 0
        v2 = self.scale > 0
        return v1 and v2
    
    def get_parameters(self, measurements):
        """
        Calculate proper parameters of the distribution from sample measurements.
        The parameters are calculated by solving the equations of the measures expected 
        for this distribution.The number of equations to consider is equal to the number 
        of parameters.
        
        Parameters
        ----------
        measurements : dict
            {"mean": *, "variance": *, "skewness": *, "kurtosis": *, "data": *}

        Returns
        -------
        parameters : dict
            {"alpha": *, "beta": *, "min": *, "max": *}
        """
        scipy_params = scipy.stats.argus.fit(measurements.data)
        parameters = {"chi": scipy_params[0], "loc": scipy_params[1], "scale": scipy_params[2]}
        return parameters
    


if __name__ == "__main__":   
    
    ## Import function to get measurements
    from measurements.measurements import MEASUREMENTS
    
    ## Import function to get measurements
    def get_data(direction):
        file  = open(direction,'r')
        data = [float(x.replace(",",".")) for x in file.read().splitlines()]
        return data
    
    ## Distribution class
    path = "../data/data_argus.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = ARGUS(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))

