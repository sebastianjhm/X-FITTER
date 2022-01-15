import scipy.stats

class T_STUDENT_3P:
    """
    T distribution
    https://en.wikipedia.org/wiki/Student%27s_t-distribution     
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.df = self.parameters["df"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    def cdf(self, x):
        """
        Cumulative distribution function
        Calculated with quadrature integration method of scipy
        """
        # print(scipy.stats.t.cdf(x, self.df, self.loc, self.scale)) 
        return scipy.stats.t.cdf((x-self.loc)/self.scale, self.df)
    
    def pdf(self, x):
        """
        Probability density function
        """
        # print(scipy.stats.t.pdf(x, self.df, self.loc, self.scale)) 
        return scipy.stats.t.pdf((x-self.loc)/self.scale, self.df)/self.scale

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
        
        scipy_params = scipy.stats.t.fit(measurements.data)
        parameters = {"df": scipy_params[0], "loc": scipy_params[1], "scale": scipy_params[2]}
        
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
    path = "../data/data_t_student_3p.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = T_STUDENT_3P(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))