import math
import random
import time
import Reporter
import numpy as np


class TSP:
    def __init__(self, matrix):
        self.distanceMatrix = matrix
        self.penalty = 5000
        self.length = self.getLength()
        self.largestPath = self.getLargestPath()
        self.possibilityMatrix = self.getPossibilityMatrix()

    def getLength(self):
        return self.distanceMatrix.shape[0]

    def getLargestPath(self):
        maxPath = 0
        for i in range(self.length):
            for j in range(self.length):
                if self.distanceMatrix[i][j] != math.inf:
                    maxPath = max(maxPath, self.distanceMatrix[i][j])
        return maxPath

    def getPossibilityMatrix(self):
        possibilityMatrix = np.zeros([self.length, self.length])
        for i in range(self.length):
            for j in range(self.length):
                if self.distanceMatrix[i][j] == 0:
                    continue
                elif self.distanceMatrix[i][j] == math.inf:
                    possibilityMatrix[i][j] = np.exp(-1 * self.largestPath / 100) / 10
                    # possibilityMatrix[i][j] = 1 / self.penalty
                else:
                    possibilityMatrix[i][j] = np.exp(-1 * self.distanceMatrix[i][j] / 100)
                    # possibilityMatrix[i][j] = 1 / self.distanceMatrix[i][j]
        return possibilityMatrix

    def calculateInfiniteNums(self):
        l = []
        for i in range(self.length):
            s = 0
            for j in range(self.length):
                if self.distanceMatrix[i][j] == math.inf:
                    s += 1
            l.append(s)
        print(l)
        return l


class Individual:
    def __init__(self, tsp, penalty, maxValue, order):
        # self.order = np.append([0],  np.random.permutation(tsp.getLength() - 1) + 1)
        self.tsp = tsp
        self.penalty = penalty
        self.order = order
        self.maxValue = maxValue
        self.penaltyOrder = 1.02

    def getOrderList(self):
        return self.order

    def fitness(self):
        distance = 0
        for i in range(self.tsp.getLength() - 1):
            d = self.tsp.distanceMatrix[self.order[i]][self.order[i + 1]]
            if d == math.inf:
                d = self.maxValue * (self.penalty ** self.penaltyOrder)
            distance += d
        d = self.tsp.distanceMatrix[self.order[self.tsp.getLength() - 1]][self.order[0]]
        if d == math.inf:
            d = self.maxValue * (self.penalty ** self.penaltyOrder)
        distance += d

        return distance

    def mutate(self):

        point1 = random.randint(0, self.tsp.getLength() - 1)
        point2 = random.randint(0, self.tsp.getLength() - 1)
        for i in range(self.tsp.getLength() // 50):
            self.order[[(point1 + i) // self.tsp.getLength(), (point2 + i) // self.tsp.getLength()]] = \
                self.order[[(point2 + i) // self.tsp.getLength(), (point1 + i) // self.tsp.getLength()]]


class EvolutionaryAlgorithm:
    def __init__(self, tsp, populationSize, nbOffsprings, mutationProbability, recombineProbability, tournament_k,
                 maxValue, debugMode=1):
        self.mutationProbability = mutationProbability
        self.recombineProbability = recombineProbability
        self.tsp = tsp
        self.populationSize = populationSize
        self.nbOffsprings = nbOffsprings
        self.tournament_k = tournament_k
        self.population = []
        self.penalty = 10
        self.debugMode = debugMode
        self.tspLength = tsp.getLength()
        self.offspringPopulation = None
        self.largestPath = tsp.distanceMatrix
        self.maxValue = maxValue
        self.initiatePopulation()

    # def initiatePopulation(self):
    #     self.population = np.array([Individual(self.tsp, self.penalty, self.maxValue)
    #                                 for i in range(self.populationSize)])

    def initiatePopulation(self):
        for times in range(self.populationSize):
            t1 = time.time()
            newOrder = np.zeros(self.tspLength, dtype=int)
            newOrder[0] = random.randint(0, self.tspLength - 1)
            notVisitedBoolList = np.ones(self.tspLength, dtype=float)
            notVisitedBoolList[newOrder[0]] = 0
            for i in range(self.tspLength - 1):
                place = newOrder[i]
                sumPossibility = np.dot(notVisitedBoolList, self.tsp.possibilityMatrix[place].T)
                newPlacePossibility = np.random.random() * sumPossibility
                sumPossibility = 0
                for j in range(self.tspLength):
                    if notVisitedBoolList[j]:
                        sumPossibility += self.tsp.possibilityMatrix[place][j]
                        if newPlacePossibility < sumPossibility:
                            newOrder[i + 1] = j
                            notVisitedBoolList[j] = 0
                            break
            newIndividual = Individual(self.tsp, self.penalty, self.maxValue, newOrder)
            s1 = (self.tspLength - 1) * self.tspLength // 2
            s2 = np.sum(newOrder)
            if s2 != s1:
                print("initialization false")
            self.population = np.append(self.population, [newIndividual])
            t2 = time.time()
            print("Initialize one individual need time: ", t2 - t1)
            self.printInfiniteValueNums(newIndividual)

    def createOffsprings(self):
        ts1 = time.time()
        self.offspringPopulation = None
        for i in range(self.nbOffsprings):
            if random.random() < self.recombineProbability:
                o1, o2 = self.orderCrossoverOperator()
                if i == 0:
                    self.offspringPopulation = [o1]
                else:
                    self.offspringPopulation = np.append(self.offspringPopulation, [o1])
                self.offspringPopulation = np.append(self.offspringPopulation, [o2])
                # self.population = np.append(self.population, [o1])
                # self.population = np.append(self.population, [o2])
                # tos2 = time.time()
                # print(" recombination time : ", tos2 - tso1)
        ts3 = time.time()
        print(" append time : ", ts3 - ts1)
        self.population = np.append(self.population, [self.offspringPopulation])
        ts2 = time.time()
        print(" create off spring time : ", ts2 - ts1)

    # def partiallyMappedCrossoverRecombination(self):
    #     p1, p2 = self.selectTwoParents()
    #     o1 = Individual(self.tsp, self.penalty, self.maxValue)
    #     o2 = Individual(self.tsp, self.penalty, self.maxValue)
    #     point = np.random.randint(0, self.tspLength - 1, size=2)
    #     point.sort()
    #     u1 = [-1 for i in range(self.tspLength)]
    #     u2 = [-1 for i in range(self.tspLength)]
    #     for i in range(point[1] - point[0] + 1):
    #         u1[p2.order[point[0] + i]] = p1.order[point[0] + i]
    #         u2[p1.order[point[0] + i]] = p2.order[point[0] + i]
    #         o1.order[point[0] + i] = p2.order[point[0] + i]
    #         o2.order[point[0] + i] = p1.order[point[0] + i]
    #
    #     def calculateOffspring(offspring, parent, usedList):
    #         for i in range(self.tspLength):
    #             if point[0] <= i <= point[1]:
    #                 continue
    #             if usedList[parent.order[i]] == -1:
    #                 usedList[parent.order[i]] = parent.order[i]
    #                 offspring.order[i] = parent.order[i]
    #             else:
    #                 nextPos = usedList[parent.order[i]]
    #                 cache = nextPos
    #                 while nextPos != -1:
    #                     cache = nextPos
    #                     nextPos = usedList[nextPos]
    #                 offspring.order[i] = cache
    #                 usedList[cache] = cache
    #         return offspring
    #
    #     o1 = calculateOffspring(o1, p1, u1)
    #     o2 = calculateOffspring(o2, p2, u2)
    #     return o1, o2

    # order crossover operator
    def orderCrossoverOperator(self):
        p1, p2 = self.selectTwoParents()
        points = random.sample(range(0, self.tspLength), 2)
        left = 0
        right = 0
        if points[0] > points[1]:
            right, left = points[0], points[1]
        else:
            right, left = points[1], points[0]
        middle1 = p2.order[left:right]
        middle2 = p1.order[left:right]
        ll1 = p1.order[right:self.tspLength]
        ll2 = p1.order[0:right]
        long1 = np.concatenate((ll1, ll2))
        ll3 = p2.order[right:self.tspLength]
        ll4 = p2.order[0:right]
        long2 = np.concatenate((ll3, ll4))

        for i in range(right - left):
            long1 = long1[long1 != middle1[i]]
            long2 = long2[long2 != middle2[i]]
        temp1 = np.concatenate((long1[self.tspLength - right:], middle1))
        temp2 = np.concatenate((long2[self.tspLength - right:], middle2))
        offspring1 = np.concatenate((temp1, long1[:self.tspLength - right]))
        offspring2 = np.concatenate((temp2, long2[:self.tspLength - right]))
        o1 = Individual(self.tsp, self.penalty, self.maxValue, offspring1)
        o2 = Individual(self.tsp, self.penalty, self.maxValue, offspring1)
        return o1, o2

    # cycle crossover operator
    # def cycleCrossoverRecombination(self):
    #     p1, p2 = self.selectTwoParents()
    #
    #     # create two offsprings
    #     o1 = Individual(self.tsp, self.penalty, self.maxValue)
    #     o2 = Individual(self.tsp, self.penalty, self.maxValue)
    #     # create index finding of two parents
    #     order1 = list(range(self.tspLength))
    #     order2 = list(range(self.tspLength))
    #     for i in range(self.tspLength):
    #         order1[p1.order[i]] = i
    #         order2[p2.order[i]] = i
    #     # used list, which means if the relevant parent nodes was detected
    #     u1 = [False for i in range(self.tspLength)]
    #     u2 = [False for i in range(self.tspLength)]
    #     index = random.randint(0, self.tspLength - 1)
    #     i = p1.order[index]
    #     while not u1[index]:
    #         o1.order[index] = i
    #         i = p2.order[index]
    #         u1[index] = True
    #         index = order1[i]
    #     for i in range(self.tspLength):
    #         if not u1[i]:
    #             o1.order[i] = p2.order[i]
    #
    #     index = 0
    #     i = p2.order[index]
    #     while not u2[index]:
    #         o2.order[index] = i
    #         i = p1.order[index]
    #         u2[index] = True
    #         index = order2[i]
    #     for i in range(self.tspLength):
    #         if not u2[i]:
    #             o2.order[i] = p1.order[i]
    #     if len(o2.order) != self.tspLength or len(o1.order) != self.tspLength:
    #         print("????????")
    #     return o1, o2

    # cycle crossover 2
    # def cycleCrossover2Recombination(self):
    #     p1, p2 = self.selectTwoParents()
    #     # create two offsprings
    #     o1 = Individual(self.tsp, self.penalty, self.maxValue)
    #     o2 = Individual(self.tsp, self.penalty, self.maxValue)
    #     # create index finding of two parents
    #     order1 = list(range(self.tspLength))
    #     order2 = list(range(self.tspLength))
    #     for i in range(self.tspLength):
    #         order1[p1.order[i]] = i
    #         order2[p2.order[i]] = i
    #     # used list, which means if the relevant parent nodes was detected
    #     u1 = [False for i in range(self.tspLength)]
    #     u2 = [False for i in range(self.tspLength)]
    #     i = 0
    #     while False not in u2:
    #         pos0 = u2.index(False)
    #         i1 = p2.order[pos0]
    #         while not u2[i1]:
    #             o1.order[i] = i1
    #             pos1 = order1[i1]
    #             i2 = p2.order[pos1]
    #             pos2 = order1[i2]
    #             i3 = p2.order[pos2]
    #             o2.order[i] = i3
    #             pos3 = order1[i3]
    #             u1[i1] = True
    #             i1 = p2.order[pos3]
    #             i = i + 1
    #     return o1, o2

    def select(self):
        players = list(np.random.choice(self.population, size=self.tournament_k))
        return min(players, key=lambda p: p.fitness())

    def selectTwoParents(self):
        p1 = self.select()
        p2 = self.select()
        t = 0
        while p1 == p2 and t < 5:
            p2 = self.select()
            t = t + 1
        return p1, p2

    def mutate(self):
        ts1 = time.time()
        for i in self.population:
            times = self.tspLength // 50
            for j in range(times):
                if random.random() < self.mutationProbability:
                    i.mutate()
        ts2 = time.time()
        print("time of mutation is : ", ts2 - ts1)

    def eliminate(self, iteration):
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
        totalFitness = 0
        realAverage = 0
        count = 0
        for i in self.population:
            temp = i.fitness()
            if temp < bestFitness:
                bestFitness = temp
                bestIndividual = i
            totalFitness = totalFitness + temp
            if count // 10 == 9:
                realAverage = realAverage + totalFitness / 10
                totalFitness = 0
            count = count + 1
        realAverage = realAverage / (self.tspLength / 10)
        return bestFitness, bestIndividual

    def printInfiniteValueNums(self, individual):
        infNums = 0
        for i in range(self.tspLength - 1):
            if self.tsp.distanceMatrix[individual.order[i]][individual.order[i + 1]] == math.inf:
                infNums += 1
        if self.tsp.distanceMatrix[individual.order[self.tspLength - 1]][individual.order[0]] == math.inf:
            infNums += 1
        print("This new individual has fitness value: ", individual.fitness())
        print("And it has ", infNums, " infinite values.")

    def __call__(self, iteration):
        self.createOffsprings()
        self.mutate()
        self.eliminate(iteration)
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
        t1 = time.time()
        tsp = TSP(distanceMatrix)
        maxValue = 0
        for i in range(tsp.distanceMatrix.shape[0]):
            for j in range(tsp.distanceMatrix.shape[1]):
                if tsp.distanceMatrix[i][j] != math.inf and maxValue < tsp.distanceMatrix[i][j]:
                    maxValue = tsp.distanceMatrix[i][j]
        t2 = time.time()
        print("find the biggest value time: ", t2 - t1)
        algorithm = EvolutionaryAlgorithm(tsp, 200, 100, 0.1, 1, 10, maxValue)
        iterationTimes = 0

        # Your code here.
        yourConvergenceTestsHere = True
        while iterationTimes < 500:
            iterationTimes += 1

            meanObjective, bestObjective, bestSolution = algorithm(iterationTimes)
            counts = 0
            for i in range(tsp.getLength() - 1):
                if tsp.distanceMatrix[bestSolution[i]][bestSolution[i + 1]] == math.inf:
                    counts = counts + 1
            if tsp.distanceMatrix[bestSolution[-1]][bestSolution[0]] == math.inf:
                counts = counts + 1
            print("best solution has ", counts, " infinite value")
            # Your code here.

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                print(timeLeft)
                break
            print(timeLeft)

        # Your code here.
        return 0


if __name__ == '__main__':
    r = r0776947()
    r.optimize("tour1000.csv")
