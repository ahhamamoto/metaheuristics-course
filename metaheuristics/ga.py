"""
Solves the binary knapsack problem using a classic Genetic Algorithm,
which was initially proposed by Holland.
"""

import numpy as np


class KnapsackGA:
    """Class that orchestrates the execution of the evolution process."""

    def __init__(self, weights, utilities, knapsack_weight,
                 population_size=50, generations=1000, mut_prob=0.03):
        """Constructor. Simply sets parameters"""
        self.weights = np.array(weights)
        self.utilities = np.array(utilities)
        self.knapsack_weight = knapsack_weight
        self.population_size = population_size
        self.generations = generations
        self.mutation_probability = mut_prob
        self.chromosome_size = self.weights.size
        self.population = None
        self.fitness = None
        self.best = np.zeros(self.chromosome_size)
        self.best_fitness = 0
        self.run()

    def run(self):
        """Runs the GA process."""
        self.generate_population()
        self.population_fitness()
        for _ in range(self.generations):
            new_population = []
            for _ in range(int(self.population_size/2)):
                mated = self.crossover(self.roulette(), self.roulette())
                new_population.append(self.mutate(mated[0]))
                new_population.append(self.mutate(mated[1]))
            self.population = new_population
            self.check_restriction()
            self.best_chromosome()
        # print(self.best)
        # print(self.best_fitness)

    def generate_population(self):
        """Randomly generates the initial population."""
        self.population = [np.random.choice([0, 1], self.chromosome_size)
                           for _ in range(self.population_size)]

    def population_fitness(self):
        """Calculates the fitness of the entire population."""
        self.fitness = np.array([(chromosome * self.utilities).sum()
                                 for chromosome in self.population])

    def roulette(self):
        """Performs the selection process using roulette wheel approach."""
        roulette_value = np.random.uniform(self.fitness.sum())
        current_value = 0
        for idx, fitness in enumerate(self.fitness):
            current_value += fitness
            if roulette_value <= current_value:
                return self.population[idx]

    def crossover(self, parent1, parent2):
        """Returns two offspring taking two parents as parameters."""
        crossover_point = np.random.randint(1, self.chromosome_size - 1, 1)[0]
        offspring1 = np.append(parent1[:crossover_point],
                               parent2[crossover_point:])
        offspring2 = np.append(parent2[:crossover_point],
                               parent1[crossover_point:])
        return offspring1, offspring2

    def mutate(self, chromosome):
        """Mutate a chromosome."""
        new_chromosome = chromosome
        for idx, gene in enumerate(chromosome):
            chance = np.random.ranf()
            if chance <= self.mutation_probability:
                new_chromosome[idx] = (gene + 1) % 2
        return new_chromosome

    def check_restriction(self):
        """Checks if chromosomes in population violate restriction. In case
        it does, it fixes the chromosome."""
        for idx, chromosome in enumerate(self.population):
            if (chromosome * self.weights).sum() > self.knapsack_weight:
                self.population[idx]  = self.fix_chromosome(chromosome)

    def fix_chromosome(self, chromosome):
        """Fixes a chromosome that violates the knapsack restriction."""
        new_chromosome = chromosome
        while ((new_chromosome * self.weights).sum() >
                   self.knapsack_weight):
            in_knapsack = [i for i, j in enumerate(new_chromosome) if j > 0]
            new_chromosome[np.random.choice(in_knapsack)] = 0
        return new_chromosome

    def best_chromosome(self):
        """Stores the best chromosome achieved."""
        i = np.argmax(self.fitness)
        if self.fitness[i] > self.best_fitness:
            self.best_fitness = self.fitness[i]
            self.best = self.population[i]
