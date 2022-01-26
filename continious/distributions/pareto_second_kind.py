import scipy.stats

class PARETO_SECOND_KIND:
    """
    Pareto second kind distribution distribution
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lomax.html
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.xm = self.parameters["xm"]
        self.alpha = self.parameters["alpha"]
        
    def cdf(self, x):
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        # print(scipy.stats.lomax.cdf(x, self.alpha, scale=self.xm))
        result = 1 - (self.xm/(x+self.xm)) ** self.alpha
        return result
    
    def pdf(self, x):
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        # print(scipy.stats.lomax.pdf(x, self.alpha, scale=self.xm))
        return (self.alpha * self.xm ** self.alpha) / ((x+self.xm) ** (self.alpha + 1))
        
    def get_num_parameters(self):
        """
        Number of parameters of the distribution
        """
        return len(self.parameters.keys())
    
    def parameter_restrictions(self):
        """
        Check parameters restriction
        """
        v1 = self.xm > 0
        v2 = self.alpha > 0
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
            {"xm": *, "alpha": *}
        """       
        ## Solve system
        m = measurements.mean
        v = measurements.variance
        
        xm = -(m * (m**2 + v))/(m**2 - v)
        alpha = -(2*v)/(m**2 - v)
        parameters = {"xm": xm , "alpha": alpha}

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
    path = "../data/data_pareto_second_kind.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = PARETO_SECOND_KIND(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(13.806))
    
    ## Get parameters of distribution: SCIPY vs EQUATIONS
    import time
    ti = time.time()
    print(distribution.get_parameters(measurements))
    print("Solve equations time: ", time.time() -ti)
    ti = time.time()
    print(scipy.stats.lomax.fit(data))
    print("Scipy time get parameters: ",time.time() - ti)

