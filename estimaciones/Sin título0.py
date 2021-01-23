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


def objective(x, p):
    A, B, C = x
    return A + B + C

def constraint_mean(x,p):
    A, B, C = x
    miu = lambda r: (A**r) * C * modificated_beta((B*C-r)/B, (B+r)/B, 1e7)
    return miu(1) - 5.4404295

def constraint_variance(x):
    A, B, C = x
    miu = lambda r: (A**r) * C * modificated_beta((B*C-r)/B, (B+r)/B, 1e7)
    return -(miu(1)**2) + miu(2) - 0.8496069

def constraint_skewness(x):
    A, B, C = x
    miu = lambda r: (A**r) * C * modificated_beta((B*C-r)/B, (B+r)/B, 1e7)
    return 2*miu(1)**3 - 3*miu(1)*miu(2) + miu(3) + 0.5

bnds = [(1,1000),(1,1000),(1,1000)]
con1 = {"type":"eq", "fun": constraint_mean, "args":(0.5,)}
con2 = {"type":"eq", "fun": constraint_variance}
con3 = {"type":"eq", "fun": constraint_skewness}

sol = minimize(objective, [1,1,1], args = (5), method="SLSQP", bounds = bnds, constraints = [con1, con2, con3])
print(sol)
