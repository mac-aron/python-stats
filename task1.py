# from dhCheck_Task1 import dhCheckCorrectness
import scipy.stats as stats
import numpy as np

def Task1(a, b, c, point1, point2, data, mu, sigma, xm, alpha, num, point3, point4, point5):
    
    prob1, prob2, MEAN_t, MEDIAN_t, MEAN_d, VARIANCE_d, prob3, prob4, ALE = 0, 0, 0, 0, 0, 0, 0, 0, 0

    # [PART 1.1] - find the probability [prob1] that the AV is -NO GREATER- than [point1]
    
    # create a triangular distribution object
    triangular_dist = stats.triang(c=(c-a)/(b-a), loc=a, scale=b-a)  # [!]
    prob1 = triangular_dist.cdf(point1)

    # [PART 1.2] - find the probability [prob2] that the AV is -GREATER- than [point2]
    prob2 = 1 - triangular_dist.cdf(point2)

    # [PART 1.3] - find [MEAN_t] and [MEDIAN_t] of the AV
    MEAN_t = triangular_dist.mean()      
    MEDIAN_t = triangular_dist.median()

    # [PART 2] - calculate the [MEAN_d] and [VARIANCE_t] of the dataset
    MEAN_d = np.mean(data)
    VARIANCE_d = np.var(data) # ddof=1

    # [PART 3.1] - randomly sample [num] points for the total impact

    # define a log-normal distribution for flaw  A
    # mu = mean of underlying normal distribution
    # sigma =  standard deviation of underlying normal distribution (must be squared)
    scale = np.exp(mu)
    lognormal_A = stats.lognorm(s=sigma, scale=scale) # [!]
 
    # define a Pareto distribution for flaw B
    # b = shape parameter 
    # xm = minimum value
    pareto_B = stats.pareto(b=alpha, scale=xm) # [!]

    # use rvs() for random variable sample
    impact_A = lognormal_A.rvs(num)
    impact_B = pareto_B.rvs(num)
    total_impact = impact_A + impact_B

    # [PART 3.2] - based on your sampling points, derive the probability [prob3] that the total impact is greater than [point3]
    prob3 = np.sum(total_impact > point3) / num
    
    # [PART 3.3] - based on your sampling points, derive the probability [prob4] that the total impact is between [point4] and [point5]
    prob4 = np.sum((total_impact > point4) & (total_impact < point5)) / num

    # [PART 4] - calculate ALE
    AV = MEDIAN_t
    ARO = MEAN_d
    EF = prob3
    ALE = ARO * (AV * EF)

    return (prob1, prob2, MEAN_t, MEDIAN_t, MEAN_d, VARIANCE_d, prob3, prob4, ALE)

if __name__ == "__main__":

    a, b, c = 10000, 35000, 18000
    point1, point2, point3, point4, point5 = 12000, 25000, 30, 50, 100
    mu, sigma, xm, alpha, num = 0, 3, 1, 4, 500000
    data = [11, 15, 9, 5, 3, 14, 16, 15, 12, 10, 11, 4, 7, 12, 6]
    
    output = Task1(a, b, c, point1, point2, data, mu, sigma, xm, alpha, num, point3, point4, point5)
    print(output)