import math
from scipy.optimize import fsolve
import scipy.stats

class JOHNSON_SU:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.xi_ = self.parameters["xi"]
        self.lambda_ = self.parameters["lambda"]
        self.gamma_ = self.parameters["gamma"]
        self.delta_ = self.parameters["delta"]
        
    def cdf(self, x):      
        """
        Cumulative distribution function
        Calculated with quadrature integration method of scipy
        """
        z = lambda x: (x-self.xi_)/self.lambda_
        result = scipy.stats.norm.cdf(self.gamma_ + self.delta_*math.asinh(z(x)))
        # result, error = scipy.integrate.quad(self.pdf, float("-inf"), x)
        return result
    
    def pdf(self, x):
        """
        Probability density function
        """
        z = lambda x: (x - self.xi_) / self.lambda_
        return (self.delta_ / (self.lambda_ * math.sqrt(2 * math.pi) * math.sqrt(z(x)**2 + 1))) * math.e ** (-(1/2) * (self.gamma_ + self.delta_ * math.asinh(z(x)))**2)
    
    def get_num_parameters(self):
        """
        Number of parameters of the distribution
        """
        return len(self.parameters.keys())
    
    def parameter_restrictions(self):
        """
        Check parameters restrictions
        """
        v1 = self.delta_ > 0
        v2 = self.lambda_ > 0
        return v1 and v2

    def get_parameters(self, measurements):
        def equations(sol_i, measurements):
            ## Variables declaration
            xi_, lambda_, gamma_, delta_ = sol_i
            
            ## Help
            w = math.exp(1 / delta_ ** 2)
            omega = gamma_ / delta_
            A = w**2 * (w**4 + 2*w**3 + 3*w**2 - 3) * math.cosh(4*omega)
            B = 4 * w**2 * (w + 2) * math.cosh(2*omega)
            C = 3*(2*w + 1)
            
            ## Parametric expected expressions
            parametric_mean = xi_ - lambda_ * math.sqrt(w) * math.sinh(omega)
            parametric_variance = (lambda_ ** 2 / 2) * (w-1) * (w * math.cosh(2*omega)+1)
            # parametric_skewness = -((lambda_ ** 3) * math.sqrt(w) * (w-1)**2 * (w * (w + 2) * math.sinh(3*omega) + 3 * math.sinh(omega)) ) / (4 * math.sqrt(parametric_variance) ** 3)
            parametric_kurtosis = ((lambda_ ** 4) * (w-1)**2 * (A+B+C)) / (8 * math.sqrt(parametric_variance) ** 4)
            parametric_median = xi_ + lambda_ * math.sinh(-omega)
            
            ## System Equations
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            eq3 = parametric_kurtosis - measurements.kurtosis
            eq4 = parametric_median - measurements.median
            
            return (eq1, eq2, eq3, eq4)
        
        solution =  fsolve(equations, (1, 1, 1, 1), measurements)
        parameters = {"xi": solution[0], "lambda": solution[1], "gamma": solution[2], "delta": solution[3]}
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
    path = "..\\data\\data_johnson_su.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = JOHNSON_SU(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    
    import time
    ti = time.time()
    print(distribution.get_parameters(measurements))
    print("Solve equations time: ",time.time() -ti)
    ti = time.time()
    print(scipy.stats.johnsonsu.fit(data))
    print("Scipy time get parameters: ",time.time() -ti)