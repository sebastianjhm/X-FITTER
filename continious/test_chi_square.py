import scipy.stats
import numpy as np
from measurements__ import MEASUREMENTS

def test_chi_square(data, distribution_class):
    """
    Chi Square test to evaluate that a sample is distributed according to a probability 
    distribution.
    
    The hypothesis that the sample is distributed following the probability distribution
    is not rejected if the test statistic is less than the critical value or equivalently
    if the p-value is less than 0.05
    
    Parameters
    ----------
    data: iterable
        data set
    distribution: class
        distribution class initialized whit parameters of distribution and methods
        cdf() and get_num_parameters()
        
    Return
    ------
    result_test_chi2: dict
        1. test_statistic(float):
            sum over all classes of the value (expected - observed) ^ 2 / expected 
        2. critical_value(float):
            inverse of the distribution chi square to 0.95 with freedom degrees
            n - 1 minus the number of parameters of the distribution.
        3. p-value([0,1]):
            right-tailed probability of the test statistic for the chi-square distribution
            with the same degrees of freedom as for the critical value calculation.
        4. rejected(bool):
            decision if the null hypothesis is rejected. If it is false, it can be 
            considered that the sample is distributed according to the probability 
            distribution. If it's true, no.
    """
    ## Init a instance of class
    measurements = MEASUREMENTS(data)
    distribution = distribution_class(measurements)

    ## Parameters and preparations
    N = measurements.length
    num_bins = measurements.num_bins
    frequencies, bin_edges = np.histogram(data, num_bins)
    freedom_degrees = num_bins - 1 - distribution.get_num_parameters()
    
    ## Calculation of errors
    errors = []
    for i, observed in enumerate(frequencies):
        lower = bin_edges[i]
        upper = bin_edges[i+1]
        expected = N * (distribution.cdf(upper) - distribution.cdf(lower))
        errors.append(((observed - expected)**2) / expected)
    
    ## Calculation of indicators
    statistic_chi2 = sum(errors)
    critical_value = scipy.stats.chi2.ppf(0.95, freedom_degrees)
    p_value = 1 -  scipy.stats.chi2.cdf(statistic_chi2, freedom_degrees)
    rejected = statistic_chi2 >= critical_value
    
    ## Construction of answer
    result_test_chi2 = {
        "test_statistic": statistic_chi2, 
        "critical_value": critical_value, 
        "p-value": p_value,
        "rejected": rejected
        }
    
    return result_test_chi2
    
if __name__ == "__main__":
    from distributions.beta import BETA
    from distributions.burr import BURR
    from distributions.burr_4p import BURR_4P
    from distributions.cauchy import CAUCHY
    from distributions.chi_square_3p import CHI_SQUARE_3P
    from distributions.chi_square import CHI_SQUARE
    from distributions.dagum import DAGUM
    from distributions.dagum_4p import DAGUM_4P
    from distributions.erlang import ERLANG
    from distributions.erlang_3p import ERLANG_3P
    from distributions.error_function import ERROR_FUNCTION
    from distributions.exponential import EXPONENTIAL
    from distributions.exponential_2p import EXPONENTIAL_2P
    from distributions.f import F
    from distributions.fatigue_life import FATIGUE_LIFE
    from distributions.frechet import FRECHET
    from distributions.gamma import GAMMA
    from distributions.gamma_3p import GAMMA_3P
    from distributions.generalized_extreme_value import GENERALIZED_EXTREME_VALUE
    from distributions.generalized_gamma import GENERALIZED_GAMMA
    from distributions.generalized_gamma_4p import GENERALIZED_GAMMA_4P
    from distributions.generalized_logistic import  GENERALIZED_LOGISTIC
    from distributions.generalized_normal import GENERALIZED_NORMAL
    from distributions.gumbel_left import GUMBEL_LEFT
    from distributions.gumbel_right import GUMBEL_RIGHT
    from distributions.hypernolic_secant import HYPERBOLIC_SECANT
    from distributions.inverse_gamma import INVERSE_GAMMA
    from distributions.inverse_gamma_3p import INVERSE_GAMMA_3P
    from distributions.inverse_gaussian import INVERSE_GAUSSIAN
    from distributions.inverse_gaussian_3p import INVERSE_GAUSSIAN_3P
    from distributions.johnson_sb import JOHNSON_SB
    from distributions.johnson_su import JOHNSON_SU
    from distributions.kumaraswamy import KUMARASWAMY
    from distributions.laplace import LAPLACE
    from distributions.levy import LEVY
    from distributions.loggamma import LOGGAMMA
    from distributions.logistic import LOGISTIC
    from distributions.loglogistic import LOGLOGISTIC
    from distributions.loglogistic_3p import LOGLOGISTIC_3P
    from distributions.lognormal import LOGNORMAL
    from distributions.nakagami import NAKAGAMI
    from distributions.normal import NORMAL
    from distributions.pareto_first_kind import PARETO_FIRST_KIND
    from distributions.pareto_second_kind import PARETO_SECOND_KIND
    from distributions.pearson_type_6 import PEARSON_TYPE_6
    from distributions.pearson_type_6_4p import PEARSON_TYPE_6_4P
    from distributions.pert import PERT
    from distributions.power_function import POWER_FUNCTION
    from distributions.rayleigh import RAYLEIGH
    from distributions.reciprocal import RECIPROCAL
    from distributions.rice import RICE
    from distributions.t import T
    from distributions.trapezoidal import TRAPEZOIDAL
    from distributions.triangular import TRIANGULAR
    from distributions.uniform import UNIFORM
    from distributions.weibull import WEIBULL
    from distributions.weibull_3p import WEIBULL_3P
    
    def get_data(direction):
        file  = open(direction,'r')
        data = [float(x.replace(",",".")) for x in file.read().splitlines()]
        return data
    
    _all_distributions = [
        BETA, BURR, CAUCHY, CHI_SQUARE, DAGUM, ERLANG, ERROR_FUNCTION, 
        EXPONENTIAL, F, FATIGUE_LIFE, FRECHET, GAMMA, GENERALIZED_EXTREME_VALUE, GENERALIZED_GAMMA_4P,
        GENERALIZED_GAMMA, GENERALIZED_LOGISTIC, GENERALIZED_NORMAL, GUMBEL_LEFT, 
        GUMBEL_RIGHT, HYPERBOLIC_SECANT, INVERSE_GAMMA, INVERSE_GAUSSIAN, JOHNSON_SB, 
        JOHNSON_SU, KUMARASWAMY, LAPLACE, LEVY, LOGGAMMA, LOGISTIC, LOGLOGISTIC,
        LOGNORMAL,  NAKAGAMI, NORMAL, PARETO_FIRST_KIND, PARETO_SECOND_KIND, PEARSON_TYPE_6, 
        PERT, POWER_FUNCTION, RAYLEIGH, RECIPROCAL, RICE, T, TRAPEZOIDAL, TRIANGULAR,
        UNIFORM, WEIBULL
    ]

    _my_distributions = [DAGUM, DAGUM_4P, POWER_FUNCTION, RICE, RAYLEIGH, RECIPROCAL, T, GENERALIZED_GAMMA_4P]
    _my_distributions = [DAGUM_4P, BURR, CHI_SQUARE_3P, EXPONENTIAL_2P, GAMMA_3P, 
                         INVERSE_GAUSSIAN_3P, LOGLOGISTIC_3P, PEARSON_TYPE_6_4P, INVERSE_GAMMA_3P, 
                         WEIBULL_3P, ERLANG_3P]
    for distribution_class in _my_distributions:
        print(distribution_class.__name__)
        path = "./data/data_" + distribution_class.__name__.lower() + ".txt"
        data = get_data(path)                
        print(test_chi_square(data, distribution_class))

    


