import math
import numpy as np

class LAPLACE:
    """
    Laplace distribution
    https://en.wikipedia.org/wiki/Laplace_distribution 
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.miu = self.parameters["miu"]
        self.b = self.parameters["b"]
        
    def cdf(self, x):
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        return 0.5 + 0.5*np.sign(x-self.miu)*(1-math.exp(-abs(x-self.miu)/self.b))
    
    def pdf(self, x):
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        return (1/(2*self.b)) * math.exp(-abs(x-self.miu)/self.b)
    
    def get_num_parameters(self):
        """
        Number of parameters of the distribution
        """
        return len(self.parameters.keys())
    
    def parameter_restrictions(self):
        """
        Check parameters restrictions
        """
        v1 = self.b > 0
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
            {"miu": *, "b": *}
        """
        miu = measurements.mean
        b = math.sqrt(measurements.variance/2)
    
        ## Results
        parameters = {"miu": miu, "b": b}

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
    path = "../data/data_laplace.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = LAPLACE(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))