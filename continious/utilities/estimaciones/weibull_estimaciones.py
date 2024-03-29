from scipy.optimize import fsolve
import math

def equations(sol_i, p):
    alpha_, beta_ = sol_i
    _mean, _variance = p
    
    ## Mean
    f1 = (beta_/alpha_) * math.gamma(1/alpha_) - _mean
    ## Variance
    f2 = (beta_**2/alpha_) * (2 * math.gamma(2/alpha_) - (1/alpha_) * math.gamma(1/alpha_)**2) - _variance
   
    return (f1, f2)

sol =  fsolve(equations, (1, 1), [9.35278469, 2.323387311])
print(sol)