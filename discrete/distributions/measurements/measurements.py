import scipy.stats
import numpy as np
import scipy.optimize
import math

class MEASUREMENTS:
    def __init__(self, data):
        self.length = len(data)
        self.min = min(data)
        self.max = max(data)
        self.mean = np.mean(data)
        self.variance = np.var(data, ddof=1)
        self.std = math.sqrt(np.var(data, ddof=1))
        self.skewness = scipy.stats.moment(data, 3) / pow(np.std(data, ddof=1),3)
        self.kurtosis = scipy.stats.moment(data, 4) / pow(np.std(data, ddof=1),4)
        self.median = np.median(data)
        self.mode = int(scipy.stats.mode(data)[0][0])
        self.frequencies = self.frequencies(data)
        self.data = data
        
    def frequencies(self, data):
        f = {}
        for x in data:
            if x in f:
                f[x] += 1
            else:
                f[x] = 1
        return({k: v for k, v in sorted(f.items(), key=lambda item: item[0])})

if __name__ == "__main__":
    ## Import function to get measurements
    def get_data(direction):
        file  = open(direction,'r')
        data = [float(x.replace(",",".")) for x in file.read().splitlines()]
        return data
    
    ## Distribution class
    path = "./data/data_binomial.txt"
    data = get_data(path) 

    measurements = MEASUREMENTS(data)

    print("Length: ", measurements.length)
    print("Min: ", measurements.min)
    print("Max: ", measurements.max)
    print("Mean: ", measurements.mean)
    print("Variance: ", measurements.variance)
    print("Skewness: ", measurements.skewness)
    print("Kurtosis: ", measurements.kurtosis)
    print("Median: ", measurements.median)
    print("Mode: ", measurements.mode)
