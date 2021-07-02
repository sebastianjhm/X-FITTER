import math
import utilities.ad_marsaglia as ad
from measurements__ import MEASUREMENTS

def test_anderson_darling(data, distribution_class):
    """
    Anderson Darling test to evaluate that a sample is distributed according to a probability 
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
    result_test_ks: dict
        1. test_statistic(float):
            sum over all data(Y) of the value ((2k-1)/N)*(ln[Fn(Y[k])]+ln[1-Fn(Y[N-k+1])]).
        2. critical_value(float):
            calculation of the Anderson Darling critical value using Marsaglia-Marsaglia function.
            whit size of sample N as parameter.
        3. p-value[0,1]:
            probability of the test statistic for the Anderson-Darling distribution
            whit size of sample N as parameter.
        4. rejected(bool):
            decision if the null hypothesis is rejected. If it is false, it can be 
            considered that the sample is distributed according to the probability 
            distribution. If it's true, no.
            
    References
    ----------
    .. [1] Marsaglia, G., & Marsaglia, J. (2004). 
           Evaluating the anderson-darling distribution. 
           Journal of Statistical Software, 9(2), 1-5.
    .. [2] Sinclair, C. D., & Spurr, B. D. (1988).
           Approximations to the distribution function of the anderson—darling test statistic.
           Journal of the American Statistical Association, 83(404), 1190-1191.
    .. [3] Lewis, P. A. (1961). 
           Distribution of the Anderson-Darling statistic. 
           The Annals of Mathematical Statistics, 1118-1124.
    """
    ## Init a instance of class
    measurements = MEASUREMENTS(data)
    distribution = distribution_class(measurements)

    ## Parameters and preparations
    N = measurements.length
    data.sort()
    
    ## Calculation S
    S = 0
    for k in range(N):
        c1 = math.log(distribution.cdf(data[k]))
        c2 = math.log(1-distribution.cdf(data[N-k-1]))
        c3 = (2*(k+1)-1)/N
        S += c3 * (c1 + c2)
    
    ## Calculation of indicators
    A2 = -N-S
    critical_value = ad.ad_critical_value(0.95, N)
    p_value = ad.ad_p_value(N, A2)
    rejected = A2 >= critical_value
    
    ## Construction of answer
    result_test_ad = {
        "test_statistic": A2, 
        "critical_value": critical_value,
        "p-value": p_value,
        "rejected": rejected
        }
    
    return result_test_ad

if __name__ == "__main__":
    from distributions.beta import BETA
    from distributions.burr import BURR
    from distributions.burr_4P import BURR_4P
    from distributions.cauchy import CAUCHY
    from distributions.chi_square import CHI_SQUARE
    from distributions.dagum import DAGUM
    from distributions.dagum_4P import DAGUM_4P
    from distributions.erlang import ERLANG
    from distributions.error_function import ERROR_FUNCTION
    from distributions.exponencial import EXPONENCIAL
    from distributions.f import F
    from distributions.fatigue_life import FATIGUE_LIFE
    from distributions.frechet import FRECHET
    from distributions.gamma import GAMMA
    from distributions.generalized_extreme_value import GENERALIZED_EXTREME_VALUE
    from distributions.generalized_gamma import GENERALIZED_GAMMA
    from distributions.generalized_gamma_4P import GENERALIZED_GAMMA_4P
    from distributions.generalized_logistic import  GENERALIZED_LOGISTIC
    from distributions.generalized_normal import GENERALIZED_NORMAL
    from distributions.gumbel_left import GUMBEL_LEFT
    from distributions.gumbel_right import GUMBEL_RIGHT
    from distributions.hypernolic_secant import HYPERBOLIC_SECANT
    from distributions.inverse_gamma import INVERSE_GAMMA
    from distributions.inverse_gaussian import INVERSE_GAUSSIAN
    from distributions.johnson_SB import JOHNSON_SB
    from distributions.johnson_SU import JOHNSON_SU
    from distributions.kumaraswamy import KUMARASWAMY
    from distributions.laplace import LAPLACE
    from distributions.levy import LEVY
    from distributions.loggamma import LOGGAMMA
    from distributions.logistic import LOGISTIC
    from distributions.loglogistic import LOGLOGISTIC
    from distributions.lognormal import LOGNORMAL
    from distributions.nakagami import NAKAGAMI
    from distributions.normal import NORMAL
    from distributions.pareto_first_kind import PARETO_FIRST_KIND
    from distributions.pareto_second_kind import PARETO_SECOND_KIND
    from distributions.pearson_type_6 import PEARSON_TYPE_6
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
    
    def get_data(direction):
        file  = open(direction,'r')
        data = [float(x.replace(",",".")) for x in file.read().splitlines()]
        return data
    
    _all_distributions = [
        BETA, BURR, CAUCHY, CHI_SQUARE, DAGUM, ERLANG, ERROR_FUNCTION, 
        EXPONENCIAL, F, FATIGUE_LIFE, FRECHET, GAMMA, GENERALIZED_EXTREME_VALUE, GENERALIZED_GAMMA_4P,
        GENERALIZED_GAMMA, GENERALIZED_LOGISTIC, GENERALIZED_NORMAL, GUMBEL_LEFT, 
        GUMBEL_RIGHT, HYPERBOLIC_SECANT, INVERSE_GAMMA, INVERSE_GAUSSIAN, JOHNSON_SB, 
        JOHNSON_SU, KUMARASWAMY, LAPLACE, LEVY, LOGGAMMA, LOGISTIC, LOGLOGISTIC,
        LOGNORMAL,  NAKAGAMI, NORMAL, PARETO_FIRST_KIND, PARETO_SECOND_KIND, PEARSON_TYPE_6, 
        PERT, POWER_FUNCTION, RAYLEIGH, RECIPROCAL, RICE, T, TRAPEZOIDAL, TRIANGULAR,
        UNIFORM, WEIBULL
    ]

    _my_distributions = [DAGUM, DAGUM_4P, POWER_FUNCTION, RICE, RAYLEIGH, RECIPROCAL, T, GENERALIZED_GAMMA_4P]
    _my_distributions = [DAGUM, DAGUM_4P, BURR_4P]
    for distribution_class in _my_distributions:
        print(distribution_class.__name__)
        path = ".\\data\\data_" + distribution_class.__name__.lower() + ".txt"
        data = get_data(path)                   
        print(test_anderson_darling(data, distribution_class))