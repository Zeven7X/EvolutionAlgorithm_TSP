import math
import random
import time
import Reporter
import numpy as np


class TSP:
    def __init__(self, matrix):
        self.distanceMatrix = matrix

    def getLength(self):
        return self.distanceMatrix.shape[0]


class Individual:
    def __init__(self, tsp):
        ts1 = time.time()
        # self.order = np.append([0],  np.random.permutation(tsp.getLength() - 1) + 1)
        self.tsp = tsp
        # print(id(tsp))
        self.order = np.random.permutation(self.tsp.getLength())


        ts2 = time.time()
        print("time of initialization is : ", ts2 - ts1)

    def fitness(self):
        distance = 0
        for i in range(self.tsp.getLength() - 1):
            d = self.tsp.distanceMatrix[self.order[i]][self.order[i + 1]]
            if d == math.inf:
                d = 50000
            distance += d
        d = self.tsp.distanceMatrix[self.order[self.tsp.getLength() - 1]][self.order[0]]
        if d == math.inf:
            d = 50000
        distance += d
        return distance

    def mutate(self):
        # for n in range(3):
        #     i = random.randint(0, self.tsp.getLength() - 1)
        #     j = random.randint(0, self.tsp.getLength() - 1)
        #     self.order[[i, j]] = self.order[[j, i]]
        i = random.randint(0, self.tsp.getLength() - 1)
        j = random.randint(0, self.tsp.getLength() - 1)
        self.order[[i, j]] = self.order[[j, i]]
    def recombinate(self, p1, p2):
        def possible_edge(table, n):
            index = np.nonzero(table[n])[0]
            if len(index) == 0:  # no edge
                return None
            if len(index) == 1:  # only one edge
                return index[0]
            if max(table[n]) == 1:  # two edges
                ind0 = len(np.nonzero(table[index[0]])[0])
                ind1 = len(np.nonzero(table[index[1]])[0])
                if ind0 > ind1:
                    return index[1]
                else:
                    return index[0]

        def update(table, n):
            for item in table:
                item[n] = 0

        l = self.tsp.getLength()
        eTable = np.zeros((l, l))  # format: eTable[element][edge]
        for i in range(l):
            eTable[p1.order[i]][p1.order[(i + 1) % l]] += 1
            eTable[p2.order[i]][p2.order[(i + 1) % l]] += 1

        remaining = list(range(self.tsp.getLength()))
        e = random.choice(remaining)

        for i in range(l):
            self.order[i] = e
            remaining.remove(e)
            update(eTable, e)
            if i == l - 1:
                break
            e = possible_edge(eTable, e)
            if e is None:
                e = random.choice(remaining)


class EvolutionaryAlgorithm:
    def __init__(self, tsp, populationSize, nbOffsprings, mutationProbability, tournament_k, debugMode = 0):
        self.mutationProbability = mutationProbability
        self.tsp = tsp
        self.populationSize = populationSize
        self.nbOffsprings = nbOffsprings
        self.tournament_k = tournament_k
        self.population = None
        self.initiatePopulation()
        self.debugMode = debugMode
    def initiatePopulation(self):

        self.population = np.array([Individual(self.tsp)
                                    for i in range(self.populationSize)])

    def createOffsprings(self):
        if self.debugMode == 0:
            ts1 = time.time()
        for i in range(self.nbOffsprings):
            p1 = self.select()
            p2 = self.select()
            o = Individual(self.tsp)
            if self.debugMode == 0:
                if i == 0:
                    ts3 = time.time()
                    print("one construction time is : ", ts3 - ts1 )
            o.recombinate(p1, p2)
            if self.debugMode == 0:
                if i == 0:
                    ts3 = time.time()
                    print("one recombination time is : ", ts3 - ts1 )
            self.population = np.append(self.population, [o])
        if self.debugMode == 0:
            ts2 = time.time()
            print("time of creating offspring is : ", ts2 - ts1)

    def select(self):
        ts1 = time.time()
        players = list(np.random.choice(self.population, size=self.tournament_k))
        return min(players, key=lambda p: p.fitness())


    def mutate(self):
        ts1 = time.time()
        for i in self.population:
            if random.random() < self.mutationProbability:
                i.mutate()
        ts2 = time.time()
        print("time of mutation is : ", ts2 - ts1)

    def eliminate(self):
        ts1 = time.time()
        p = list(self.population)
        p.sort(key=lambda p: p.fitness())
        p = p[:self.populationSize]
        self.population = np.array(p)
        ts2 = time.time()
        print("time of elimination is : ", ts2 - ts1)

    def getMeanFitness(self):
        return np.mean([i.fitness() for i in self.population])

    def getBestFitness(self):
        bestFitness = self.population[0].fitness()
        bestIndividual = self.population[0]
        for i in self.population:
            if i.fitness() < bestFitness:
                bestFitness = i.fitness()
                bestIndividual = i
        return bestFitness, bestIndividual

    def __call__(self):
        self.createOffsprings()
        self.mutate()
        self.eliminate()

        mean = self.getMeanFitness()
        bestFitness, bestIndividual = self.getBestFitness()
        return mean, bestFitness, bestIndividual.order


# Modify the class name to match your student number.
class r0776947:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        tsp = TSP(distanceMatrix)
        algorithm = EvolutionaryAlgorithm(tsp, 100, 100, 0.1, 10)
        i = 0

        # Your code here.
        yourConvergenceTestsHere = True
        while i < 500:
            i += 1

            meanObjective, bestObjective, bestSolution = algorithm()

            # Your code here.

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break

        # Your code here.
        return 0


if __name__ == '__main__':
    r = r0776947()
    r.optimize("tour1000.csv")
