import scipy.integrate
from scipy.optimize import minimize
from scipy.optimize import least_squares
from scipy.special import beta
import numpy as np
import scipy.stats
import warnings
+
-warnings.filterwarnings("ignore")

class BURR:
    """
    Burr distribution
    Conpendium.pdf pg.27
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.A = self.parameters["A"]
        self.B = self.parameters["B"]
        self.C = self.parameters["C"]
        
    def cdf(self, x):
        """
        Cumulative distribution function.
        Calculated with quadrature integration method of scipy.
        """
        return 1 - ((1 + (x/self.A) ** (self.B )) ** (-self.C))
    
    def pdf(self, x):
        """
        Probability density function
        """
        return ((self.B * self.C)/self.A) * ((x/self.A) ** (self.B - 1)) * ((1 + (x/self.A) ** (self.B )) ** (-self.C - 1))
        
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
            {"a": * , "b": *, "c": *}
        """
        
        # def equations(sol_i, measurements):
        #     ## Variables declaration
        #     A, B, C = sol_i
            
        #     ## Moments Burr Distribution
        #     miu = lambda r: (A**r) * C * beta((B*C-r)/B, (B+r)/B)
            
        #     ## Parametric expected expressions
        #     parametric_mean = miu(1)
        #     parametric_variance = -(miu(1)**2) + miu(2)
        #     # parametric_skewness = 2*miu(1)**3 - 3*miu(1)*miu(2) + miu(3)
        #     # parametric_kurtosis = -3*miu(1)**4 + 6*miu(1)**2 * miu(2) -4 * miu(1) * miu(3) + miu(4)
        #     parametric_median = A * ((2**(1/C))-1)**(1/B)
            
        #     ## System Equations
        #     eq1 = parametric_mean - measurements["mean"]
        #     eq2 = parametric_variance - measurements["variance"]
        #     eq3 = parametric_median - measurements["median"]
        
        #     return (eq1, eq2, eq3)
        
        # x0 = [measurements["mean"], measurements["mean"], measurements["mean"]]
        # b = ((1, 1, 1), (np.inf, np.inf, np.inf))
        # solution = least_squares(equations, x0, bounds = b, args=([measurements]))
        # parameters = {"A": solution.x[0], "B": solution.x[1], "C": solution.x[2]}
        
        scipy_params = scipy.stats.burr12.fit(measurements["data"])
        parameters = {"A": scipy_params[3], "B": scipy_params[0], "C": scipy_params[1]}
        return parameters
    
    
# def get_measurements(data: list) -> dict:
#     import scipy.stats
#     import numpy as np
#     measurements = {}
    
#     miu_3 = scipy.stats.moment(data, 3)
#     miu_4 = scipy.stats.moment(data, 4)
#     mean = np.mean(data)
#     variance = np.var(data, ddof=1)
#     skewness = miu_3 / pow(np.std(data, ddof=1),3)
#     kurtosis = miu_4 / pow(np.std(data, ddof=1),4)
#     median = np.median(data)
#     mode = scipy.stats.mode(data)[0][0]
    
#     measurements["mean"] = mean
#     measurements["variance"] =  variance
#     measurements["skewness"] = skewness
#     measurements["kurtosis"] = kurtosis
#     measurements["data"] = data
#     measurements["median"] = median
#     measurements["mode"] = mode
    
#     return measurements

# def getData(direction):
#     file  = open(direction,'r')
#     data = [float(x.replace(",",".")) for x in file.read().splitlines()]
#     return data

# path = "C:\\Users\\USUARIO1\\Desktop\\Fitter\\data\\data_burr.txt"
# data = getData(path) 
# measurements = get_measurements(data)
# distribution = BURR(measurements)
# print(distribution.get_parameters(measurements))





# def equations(sol_i, measurements, infinite):
#     ## Variables declaration
#     A, B, C = sol_i
    
#     ## Moments Burr Distribution
#     miu = lambda r: (A**r) * C * modificated_beta(abs((B*C-r)/B), abs((B+r)/B), infinite)
    
#     ## Parametric expected expressions
#     parametric_mean = miu(1)
#     parametric_variance = -(miu(1)**2) + miu(2)
#     parametric_skewness = 2*miu(1)**3 - 3*miu(1)*miu(2) + miu(3)
#     # parametric_kurtosis = -3*miu(1)**4 + 6*miu(1)**2 * miu(2) -4 * miu(1) * miu(3) + miu(4)
#     # parametric_median = A * ((2**(1/C))-1)**(1/B)
    
#     ## System Equations
#     eq1 = parametric_mean - measurements["mean"]
#     eq2 = parametric_variance - measurements["variance"]
#     eq3 = parametric_skewness - measurements["skewness"]
#     return (eq1, eq2, eq3)

# def get_parameters(self, measurements):
#         """
#         Calculate proper parameters of the distribution from sample measurements.
#         The parameters are calculated by formula.
        
#         Parameters
#         ----------
#         measurements : dict
#             {"mean": *, "variance": *, "skewness": *, "kurtosis": *, "data": *}

#         Returns
#         -------
#         parameters : dict
#             {"a": * , "b": *, "c": *}
#         """
#         def modificated_beta(x, y, infinite):
#             if beta(x,y) == float('inf'):
#                 return infinite
#             elif beta(x,y) == float('-inf'):
#                 return -infinite
#             else:
#                 return beta(x,y)
        
#         def objective(x):
#             A, B, C = x
#             return A + B + C
        
#         def constraint_mean(x, inf):
#             A, B, C = x
#             miu = lambda r: (A**r) * C * modificated_beta((B*C-r)/B, (B+r)/B, inf)
#             return miu(1) - measurements["mean"]
        
#         def constraint_variance(x, inf):
#             A, B, C = x
#             miu = lambda r: (A**r) * C * modificated_beta((B*C-r)/B, (B+r)/B, inf)
#             return -(miu(1)**2) + miu(2) - measurements["variance"]
        
#         def constraint_skewness(x, inf):
#             A, B, C = x
#             miu = lambda r: (A**r) * C * modificated_beta((B*C-r)/B, (B+r)/B, inf)
#             return 2*miu(1)**3 - 3*miu(1)*miu(2) + miu(3) - measurements["skewness"]
        
        
#         def solution_error(parameters, measurements):
#             A, B, C = parameters["A"], parameters["B"], parameters["C"]

#             miu = lambda r: (A**r) * C * beta((B*C-r)/B, (B+r)/B)
            
#             # Parametric expected expressions
#             parametric_mean = miu(1)
#             parametric_variance = -(miu(1)**2) + miu(2)
#             parametric_kurtosis = -3*miu(1)**4 + 6*miu(1)**2 * miu(2) -4 * miu(1) * miu(3) + miu(4)
#             parametric_skewness = 2*miu(1)**3 - 3*miu(1)*miu(2) + miu(3)
#             parametric_median = A * ((2**(1/C))-1)**(1/B)
                        
#             error1 = parametric_mean - measurements["mean"]
#             error2 = parametric_variance - measurements["variance"]
#             error3 = parametric_skewness - measurements["skewness"]
#             error4 = parametric_kurtosis - measurements["kurtosis"]
#             error5 = parametric_median - measurements["median"]
            
#             total_error = abs(error1) + abs(error2) + abs(error3) + abs(error4) + abs(error5)
#             return total_error
            
#         min_data = min(measurements["data"])
#         max_data = max(measurements["data"])
#         orders = [1e5, 1e6, 1e7, 1e8]
#         contextual_orders = [o * min_data for o in orders] + [o * max_data for o in orders] 
  
#         min_error = float("inf")
#         for inf in contextual_orders:
#             bnds = [(1, max_data * 100),(1, max_data * 100),(1, max_data * 100)]
#             con1 = {"type":"eq", "fun": constraint_mean, "args": (inf,)}
#             con2 = {"type":"eq", "fun": constraint_variance, "args": (inf,)}
#             con3 = {"type":"eq", "fun": constraint_skewness, "args": (inf,)}
            
#             solution = minimize(objective, [1, 1, 1], method="SLSQP", bounds = bnds, constraints = [con1, con2, con3])
#             partial_parameters = {"A": solution.x[0], "B": solution.x[1], "C": solution.x[2]}
            
#             if solution_error(partial_parameters, measurements) < min_error:
#                 min_error = solution_error(partial_parameters, measurements)
#                 parameters = partial_parameters        
                
#         print(parameters)
#         return parameters