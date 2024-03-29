from scipy.optimize import least_squares
from scipy.special import beta
import numpy as np

def modificated_beta(x, y, infinite):
    if beta(x,y) == float('inf'):
        return infinite
    elif beta(x,y) == float('-inf'):
        return -infinite
    else:
        return beta(x,y)

def equations(sol_i, data_mean, data_variance, data_median):
    ## Variables declaration
    A, B, C = sol_i
    
    ## Moments Burr Distribution
    miu = lambda r: (A**r) * C * beta((B*C-r)/B, (B+r)/B)
    
    ## Parametric expected expressions
    parametric_mean = miu(1)
    parametric_variance = -(miu(1)**2) + miu(2)
    # parametric_skewness = 2*miu(1)**3 - 3*miu(1)*miu(2) + miu(3)
    # parametric_kurtosis = -3*miu(1)**4 + 6*miu(1)**2 * miu(2) -4 * miu(1) * miu(3) + miu(4)
    parametric_median = A * ((2**(1/C))-1)**(1/B)
    
    ## System Equations
    eq1 = parametric_mean - data_mean
    eq2 = parametric_variance - data_variance
    eq3 = parametric_median - data_median

    return (eq1, eq2, eq3)

#############################################################################
## Import function to get measurements
import sys
sys.path.append("C:/Users/USUARIO/Desktop/Fitter/utilities")
from measurements_cont.measurements import MEASUREMENTS

## Import function to get measurements
def get_data(direction):
    file  = open(direction,'r')
    data = [float(x.replace(",",".")) for x in file.read().splitlines()]
    return data

## Distribution class
path = "C:/Users/USUARIO/Desktop/Fitter/data/data_burr.txt"
data = get_data(path) 
measurements = MEASUREMENTS(data)
#############################################################################

solution = least_squares(equations, (8, 8, 8), bounds = ((1, 1, 1), (np.inf, np.inf, np.inf)), args=(measurements.mean, measurements.variance, measurements.median))
parameters = {"A": solution.x[0], "B": solution.x[1], "C": solution.x[2]}
print(parameters)








def solution_error(parameters, measurements):
    A, B, C = parameters["A"], parameters["B"], parameters["C"]

    miu = lambda r: (A**r) * C * beta((B*C-r)/B, (B+r)/B)
    
    # Parametric expected expressions
    parametric_mean = miu(1)
    parametric_variance = -(miu(1)**2) + miu(2)
    parametric_kurtosis = -3*miu(1)**4 + 6*miu(1)**2 * miu(2) -4 * miu(1) * miu(3) + miu(4)
    parametric_skewness = 2*miu(1)**3 - 3*miu(1)*miu(2) + miu(3)
    parametric_median = A * ((2**(1/C))-1)**(1/B)
    print(parametric_mean, parametric_variance, parametric_median)
    print(measurements.mean, measurements.variance, measurements.median)
    
    error1 = parametric_mean - measurements.mean
    error2 = parametric_variance - measurements.variance
    error3 = parametric_skewness - measurements.skewness
    error4 = parametric_kurtosis - measurements.kurtosis
    error5 = parametric_median - measurements.median
    
    total_error = abs(error1) + abs(error2) + abs(error3) + abs(error4) + abs(error5)
    return total_error
print(solution_error(parameters, measurements))