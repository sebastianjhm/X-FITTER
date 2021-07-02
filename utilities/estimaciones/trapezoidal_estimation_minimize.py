from scipy.optimize import minimize

def objective(sol_i, p):
    b, c = sol_i
    return b + c

def constraint_mean(sol_i, data_mean, a, d):
    b, c = sol_i
    parametric_mean = (1/(3*(d + c - a - b))) * ((d**3-c**3)/(d-c) - (b**3-a**3)/(b-a))
    return parametric_mean - data_mean

def constraint_variance(sol_i, data_variance, a, d):
    b, c = sol_i
    parametric_variance = (1/(6*(d + c - a - b))) * ((d**4-c**4)/(d-c) - (b**4-a**4)/(b-a)) - ((1/(3*(d + c - a - b))) * ((d**3-c**3)/(d-c) - (b**3-a**3)/(b-a)))**2
    return parametric_variance - data_variance

def constraint(sol_i):
    b, c = sol_i
    return c - b

bnds = [(1,1000),(1,1000)]
con1 = {"type":"eq", "fun": constraint_mean, "args":(504.76, 100, 1000)}
con2 = {"type":"eq", "fun": constraint_variance, "args":(44997.32, 100, 1000)}
con3 = {"type":"ineq", "fun": constraint}

sol = minimize(objective, [1100/4,0.75*1100], args = ("parameter_objective"), method="SLSQP", bounds = bnds, constraints = [con1, con2, con3])
print(sol)
