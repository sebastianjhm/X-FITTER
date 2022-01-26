import math
import scipy.stats

class ALPHA:
    """
    Alpha distribution
    http://bayanbox.ir/view/5343019340232060584/Norman-L.-Johnson-Samuel-Kotz-N.-Balakrishnan-BookFi.org.pdf          
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha = self.parameters["alpha"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]
        
    def cdf(self, x):
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        # print(scipy.stats.alpha.cdf(x, self.alpha, loc=self.loc, scale=self.scale))
        z = lambda x: (x - self.loc)/self.scale
        result = scipy.stats.norm.cdf(self.alpha - 1/z(x))/scipy.stats.norm.cdf(self.alpha)
        return result
    
    def pdf(self, x):
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        # print(scipy.stats.alpha.pdf(x, self.alpha, loc=self.loc, scale=self.scale))
        z = lambda x: (x - self.loc)/self.scale
        result = (1/(self.scale*z(x)*z(x)*scipy.stats.norm.cdf(self.alpha)*math.sqrt(2*math.pi))) * math.exp(-0.5*(self.alpha - 1/z(x))**2)
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
        v1 = self.alpha > 0
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
            {"mean": *, "variance": *, "skewness": *, "kurtosis": *, "median": *, "mode": *}

        Returns
        -------
        parameters : dict
            {"alpha": *, "beta": *, "min": *, "max": *}
        """
        scipy_params = scipy.stats.alpha.fit(measurements.data)
        parameters = {"alpha": scipy_params[0], "loc": scipy_params[1], "scale": scipy_params[2]}
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
    path = "../data/data_alpha.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = ALPHA(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))