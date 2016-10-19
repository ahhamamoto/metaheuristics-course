"""
Script to test the implemented metaheuristics. The example present is taken
from the textbook.
"""

import numpy as np
from tabulate import tabulate

from metaheuristics import heuristic

# weight, utilities and knapsack weight globally set
weights = np.array([5, 4, 4, 2, 4, 6, 10, 4, 2, 8, 12, 5])
utilities = np.array([3, 2, 8, 4, 6, 4, 12, 2, 6, 10, 15, 9])
knapsack_weight = 36


def test_heuristic():
    """Tests the heuristic approach."""
    solution = heuristic.knapsack(weights, utilities, knapsack_weight)
    total_utility = (solution * utilities).sum()
    total_weight = (solution * weights).sum()

    print(tabulate([[solution, total_weight, total_utility]],
                   headers=['Solution', 'Weight', 'Utility'],
                   tablefmt='grid'))


def test():
    """Tests all approaches developed."""
    heuristic_solution = heuristic.knapsack(weights, utilities, knapsack_weight)
    heuristic_weight = (heuristic_solution * weights).sum()
    heuristic_utility = (heuristic_solution * utilities).sum()

    headers = ['Method', 'Solution', 'Utility',
               'Total weight / Knapsack weight']
    to_print = [
        ['Heuristic', heuristic_solution, heuristic_utility,
         str(heuristic_weight) + '/' + str(knapsack_weight)],
    ]

    print(tabulate(to_print, headers=headers, tablefmt='grid'))


if __name__ == '__main__':
    test()
