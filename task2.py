# from dhCheck_Task2 import dhCheckCorrectness
import numpy as np

def _toList(input):
    array = []
    if isinstance(input, int):
        # add the digit to a list
        array.append(input)
    elif type(input) == list:
        # keep the list as is
        array = input
    return array

def Task2(num, table, eventA, eventB, probs):
    prob1, prob2, IsInd, prob3, prob4 = 0, 0, 0, 0, 0

    # find the total number of observations in the table 
    total_cases = sum(sum(row) for row in table)

    # turn every every event into a list 
    eventA_array = _toList(eventA)
    eventB_array = _toList(eventB)

    # [TASK 1.1] - calculate the [prob1] of [eventA], the input is column(s)
    eventA_cases = 0
    intersection_sum = 0
    for column_index in eventA_array:
        for row in table:
            eventA_cases += row[column_index]
        for row_index in eventB_array:
            intersection_sum += table[row_index][column_index]
    prob1 = eventA_cases / total_cases

    # [TASK 1.2] - calculate the [prob2] of [eventB], the input is rows(s)
    eventB_cases = 0
    for row_index in eventB_array:
        eventB_cases += sum(table[row_index])
    prob2 = eventB_cases / total_cases
    
    # [TASK 1.3] - determine whether these two events are independent, [1] yes (independent), [0] no (dependent)

    # independent if (P(A|B) = P(A) & P(B|A) = P(B))
    P_A_given_B = (intersection_sum / total_cases) / prob2
    P_B_given_A = (intersection_sum / total_cases) / prob1

    if (P_A_given_B == prob1) & (P_B_given_A == prob2): IsInd = 1
    else: IsInd = 0

    # [TASK 2.1] - find the probability [prob3] of being tested positive

    # NOTE: the labels (Y = 3,4,5) for now changed to X,Y,Z and (X = 5,6,7,8) is now A,B,C,D

    # P(T) = P(T|X) * P(X) + P(T|Y) * P(Y) + P(T|Z) * P(Z) + P(T|A) * P(A) + P(T|B) * P(B) + P(T|C) * P(C)
    # to calculate P(T) use the law of total probability, which 
    # can be calculated by summing the probabilities of the event given each 
    # possible condition, weighted by the probabilities of each condition
    P_T_given_X = probs[0]
    P_T_given_Y = probs[1]
    P_T_given_Z = probs[2]
    P_T_given_A = probs[3]
    P_T_given_B = probs[4]
    P_T_given_C = probs[5]

    # probability of every row
    P_X = sum(table[0])/total_cases
    P_Y = sum(table[1])/total_cases
    P_Z = sum(table[2])/total_cases
    
    # probability of every collumn
    A_sum, B_sum, C_sum, D_sum = 0, 0, 0, 0
    for row in table:
        A_sum += row[0]
        B_sum += row[1]
        C_sum += row[2]
        D_sum += row[3]
    P_A = A_sum / total_cases
    P_B = B_sum / total_cases
    P_C = C_sum / total_cases
    P_D = D_sum / total_cases

    # calculate the probability of a positive test [prob3]
    P_T = (P_T_given_X * P_X) + (P_T_given_Y * P_Y) + (P_T_given_Z * P_Z) # + (P_T_given_A * P_A) + (P_T_given_B * P_B) + (P_T_given_C * P_C)
    prob3 = P_T

    # [TASK 2.2] - find the probability [prob4] of X=8 given that a case is tested positive  
    P_T_given_D = (prob3 - ((P_A*P_T_given_A) + (P_B*P_T_given_B) + (P_C*P_T_given_C))) / P_D
    prob4 = P_T_given_D * P_D / prob3
    
    return (prob1, prob2, IsInd, prob3, prob4)

if __name__ == "__main__":

    num = 120
    table = [[6, 10, 11, 9], 
             [9, 12, 15, 8], 
             [7, 14, 10, 9]]
    eventA = 2      # column 2 of table
    eventB = [0, 1] # row 0 and row 1 of table 
    probs = [0.7, 0.6, 0.5, 0.63, 0.44, 0.36]
    output = Task2(num, table, eventA, eventB, probs)
    print(output)
