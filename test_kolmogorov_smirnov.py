import scipy.stats

def test_kolmogorov_smirnov(data, distribution):
    """
    Kolmogorov Smirnov test to evaluate that a sample is distributed according to a probability 
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
            sum over all data of the value |Sn-Fn|
        2. critical_value(float):
            inverse of the kolmogorov-smirnov distribution to 0.95 whit size of 
            sample N as parameter.
        3. p-value[0,1]:
            probability of the test statistic for the kolmogorov-smirnov distribution
            whit size of sample N as parameter.
        4. rejected(bool):
            decision if the null hypothesis is rejected. If it is false, it can be 
            considered that the sample is distributed according to the probability 
            distribution. If it's true, no.
    """
    ## Parameters and preparations
    N = len(data)
    data.sort()
    
    ## Calculation of errors
    errors = []
    for i in range(N):
        Sn = (i + 1) / N
        if i < N - 1:
            if (data[i] != data[i+1]):
                Fn = distribution.cdf(data[i])
                errors.append(abs(Sn - Fn))
            else:
                Fn = 0
        else:
            Fn = distribution.cdf(data[i])
            errors.append(abs(Sn - Fn))
    
    ## Calculation of indicators
    statistic_ks = max(errors)
    critical_value = scipy.stats.kstwo.ppf(0.95, N)
    p_value = 1 -  scipy.stats.kstwo.cdf(statistic_ks, N)
    rejected = statistic_ks >= critical_value
    
    ## Construction of answer
    result_test_ks = {
        "test_statistic": statistic_ks, 
        "critical_value": critical_value, 
        "p-value": p_value,
        "rejected": rejected
        }
    
    return result_test_ks
    
if __name__ == "__main__":
    from utilities.data_measurements import get_measurements
    from distributions.beta import BETA
    from distributions.burr import BURR
    from distributions.cauchy import CAUCHY
    from distributions.chi_square import CHI_SQUARE
    from distributions.dagum import DAGUM
    from distributions.erlang import ERLANG
    from distributions.error_function import ERROR_FUNCTION
    from distributions.exponencial import EXPONENCIAL
    from distributions.f import F
    from distributions.fatigue_life import FATIGUE_LIFE
    from distributions.frechet import FRECHET
    from distributions.gamma import GAMMA
    from distributions.generalized_normal import GENERALIZED_NORMAL
    from distributions.johnson_SB import JOHNSON_SB
    from distributions.johnson_SU import JOHNSON_SU
    from distributions.lognormal import LOGNORMAL
    from distributions.normal import NORMAL
    from distributions.triangular import TRIANGULAR
    from distributions.uniform import UNIFORM
    from distributions.weibull import WEIBULL
    
    
    def getData(direction):
        file  = open(direction,'r')
        data = [float(x.replace(",",".")) for x in file.read().splitlines()]
        return data
    
    _all_distributions = [BETA, BURR, CAUCHY, CHI_SQUARE, DAGUM, ERLANG, ERROR_FUNCTION, EXPONENCIAL, F, FATIGUE_LIFE, FRECHET, GAMMA, GENERALIZED_NORMAL, JOHNSON_SB, JOHNSON_SU, LOGNORMAL, NORMAL, TRIANGULAR, UNIFORM,  WEIBULL]
    
    for distribution_class in _all_distributions:
        print(distribution_class.__name__)
        path = "C:\\Users\\USUARIO1\\Desktop\\Fitter\\data\\data_" + distribution_class.__name__.lower() + ".txt"
        data = getData(path)
                
        measurements = get_measurements(data)
        distribution = distribution_class(measurements)
                
        print(test_kolmogorov_smirnov(data, distribution))
