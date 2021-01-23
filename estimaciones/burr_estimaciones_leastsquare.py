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

def equations(sol_i):
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
    eq1 = parametric_mean - 19.45507362
    eq2 = parametric_variance - 5.997085691
    eq3 = parametric_median - 19.66113464

    return (eq1, eq2, eq3)

res = least_squares(equations, (19, 19, 19), bounds = ((1, 1, 1), (np.inf, np.inf, np.inf)))
print(res)