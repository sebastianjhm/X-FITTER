from scipy.optimize import minimize
import math
from scipy.special import beta

def modificated_beta(x, y, infinite):
    if beta(x,y) == float('inf'):
        return infinite
    elif beta(x,y) == float('-inf'):
        return -infinite
    else:
        return beta(x,y)


def objective(sol_i, p):
    A, B, C = sol_i
    return A + B + C

def constraint_mean(sol_i, data_mean):
    A, B, C = sol_i
    miu = lambda r: (A**r) * C * modificated_beta((B*C-r)/B, (B+r)/B, 1e7)
    return miu(1) - data_mean

def constraint_variance(sol_i, data_variance):
    A, B, C = sol_i
    miu = lambda r: (A**r) * C * modificated_beta((B*C-r)/B, (B+r)/B, 1e7)
    return -(miu(1)**2) + miu(2) - data_variance

def constraint_median(sol_i, data_median):
    A, B, C = sol_i
    miu = lambda r: (A**r) * C * modificated_beta((B*C-r)/B, (B+r)/B, 1e7)
    return A * ((2**(1/C))-1)**(1/B) - data_median

bnds = [(1,1000),(1,1000),(1,1000)]
con1 = {"type":"eq", "fun": constraint_mean, "args":(19.45,)}
con2 = {"type":"eq", "fun": constraint_variance, "args":(5.99,)}
con3 = {"type":"eq", "fun": constraint_median, "args":(19.66,)}

sol = minimize(objective, [1,1,1], args = ("parameter_objective"), method="SLSQP", bounds = bnds, constraints = [con1, con2, con3])
print(sol)
