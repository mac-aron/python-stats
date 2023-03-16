# from dhCheck_Task3 import dhCheckCorrectness
import numpy as np
from math import floor
from scipy.optimize import linprog

def Task3(x, y, z, num1, num2, num3, num4, bound_y, bound_z, c, se_bound, ml_bound, x_bound, x_initial):
    
    weights_b, weights_d, s_num5, l_num5, x_add = 0, 0, 0, 0, []

    # [TASK 1] - derive the underlying [weights_b] and [weights_d]
    X_array = np.array(x)
    y_array = np.array(y)
    z_array = np.array(z)

    # set up the regressiom with the X data 
    X_regression = np.concatenate((np.ones((X_array.shape[1],1)), X_array.T), axis=1)

    # calculate the coefficients of the linear functions using ordinary least squares (OLS)
    weights_b = np.linalg.lstsq(X_regression, y_array, rcond=None)[0]
    weights_d = np.linalg.lstsq(X_regression, z_array, rcond=None)[0]
    
    # [TASK 2] - find the smallest value [s_num5] and the largest value [l_num5] of [x5]
    provided_x = np.array([1, num1, num2, num3, num4])
    s_num5 = floor((bound_z - np.dot(weights_d[:-1], provided_x)) / weights_d[-1])
    l_num5 = floor((bound_y - np.dot(weights_b[:-1], provided_x)) / weights_b[-1])
    
    # [TASK 3]
    # linear programming constraints
    A_ub = np.vstack([-weights_b[1:], weights_d[1:]]) # [OK]
    b_ub_1 = np.array([-se_bound + weights_b[0], ml_bound - weights_d[0]]) 
    bounds = [(x_initial[i],x_bound[i]) for i in range(len(x_initial))] # [OK]

    # solve the linear programming problem
    result = linprog(c, A_ub=A_ub, b_ub=b_ub_1, A_eq=None, b_eq=None, bounds=bounds)
    x_add = (np.around(result.x - x_initial))

    # weights and x_add outputs must be with dots! no commas, and no array() when using a print()
    return (weights_b, weights_d, s_num5, l_num5, x_add)

if __name__ == "__main__":
    x= [[5,4,8,8,2,5,5,7,8,8],
        [3,7,7,2,2,5,10,4,6,3],
        [8,3,6,7,9,10,6,2,2,3],
        [9,3,9,3,10,4,2,3,7,5],
        [4,9,6,6,10,3,8,8,4,6]]
    y = [176,170,215,146,228,145,183,151,160,151]
    z = [352,384,471,358,412,345,449,357,366,349]

    num1, num2, num3, num4, bound_y, bound_z = 5, 6, 8, 4, 160, 600

    c = [11, 6, 8, 10, 9]
    se_bound = 1000
    ml_bound = 2000
    x_bound = [30,50,20,45,50]
    x_initial = [3,5,4,2,1]

    output = Task3(x, y, z, num1, num2, num3, num4, bound_y, bound_z, c, se_bound, ml_bound, x_bound, x_initial)
    print(output)