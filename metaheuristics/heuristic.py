"""
Solves the binary knapsack problem using a heuristic based on the
utility/weight ratio of the available elements.
"""

import numpy as np


def knapsack(weights, utilities, knapsack_weight):
    """
    Uses a greedy algorithm to solve the binary knapsack problem:
    1. Order the utility/weight (descendent)
    2. Put item in bag if it fits, else go to next item
    """
    weights, utilities = np.array(weights), np.array(utilities)
    number_of_items = weights.size

    benefits = utilities / weights

    ordered_indexes = np.argsort(benefits)[::-1]

    solution = np.zeros(number_of_items)

    # puts the item in the bag and if it excedes the weight, remove it from
    # bag and goes to the next iteration
    for i in ordered_indexes:
        solution[i] = 1
        if (solution * weights).sum() > knapsack_weight:
            solution[i] = 0

    return solution
