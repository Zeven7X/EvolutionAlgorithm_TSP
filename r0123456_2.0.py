import math
import random
import time
import Reporter
import numpy as np
import matplotlib.pyplot as plt


class TSP:
    def __init__(self, matrix):
        self.distanceMatrix = matrix
        self.penalty = 5000
        self.length = self.getLength()
        self.largestPath = self.getLargestPath()
        self.possibilityMatrix = self.getPossibilityMatrix()

    def getMaxValue(self):
        maxValue = 0
        for i in range(self.distanceMatrix.shape[0]):
            for j in range(self.distanceMatrix.shape[1]):
                if self.distanceMatrix[i][j] != math.inf and maxValue < self.distanceMatrix[i][j]:
                    maxValue = self.distanceMatrix[i][j]
        return maxValue

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
                    possibilityMatrix[i][j] = np.exp(-1 * self.largestPath / 9)
                    # possibilityMatrix[i][j] = 1 / self.penalty
                else:
                    possibilityMatrix[i][j] = np.exp(-1 * self.distanceMatrix[i][j] / 9)
                    # possibilityMatrix[i][j] = 1 / self.distanceMatrix[i][j]\
        print(possibilityMatrix)
        return possibilityMatrix


class Individual:
    def __init__(self, tsp, penalty, maxValue, order):
        # self.order = np.append([0],  np.random.permutation(tsp.getLength() - 1) + 1)
        self.tsp = tsp
        self.penalty = penalty
        self.order = order
        self.maxValue = maxValue
        self.penaltyOrder = 1.02
        self.fitnessValue = self.fitness()
        self.adjacency = self.getAdjacencyRepresentation()

    def getAdjacencyRepresentation(self):
        adjacencyList = np.zeros(self.tsp.getLength(), dtype=int)
        for i in range(-1, self.tsp.getLength() - 1, 1):
            adjacencyList[self.order[i]] = self.order[i + 1]
        return adjacencyList

    def getOrderList(self):
        return self.order

    def fitness(self):
        distance = 0
        for i in range(-1, self.tsp.getLength() - 1, 1):
            distance += self.maxValue * (self.penalty ** self.penaltyOrder) \
                if self.tsp.distanceMatrix[self.order[i]][self.order[i + 1]] == math.inf \
                else self.tsp.distanceMatrix[self.order[i]][self.order[i + 1]]
        return distance

    def swapMutation(self):
        points = random.sample(range(0, self.tsp.getLength()), 2)

        self.order[points[0]], self.order[points[1]] = self.order[points[1]], self.order[points[0]]

        self.fitnessValue = self.fitness()
        self.adjacency = self.getAdjacencyRepresentation()

    def replaceOrder(self, newOrder, newFitness=-1):
        self.order = newOrder.copy()
        if newFitness != -1:
            self.fitnessValue = newFitness
        else:
            self.fitnessValue = self.fitness()
            self.adjacency = self.getAdjacencyRepresentation()

    def findLongestPath(self):
        minLength = math.inf
        minIndex = 0
        for i in range(-1, self.tsp.getLength() - 1, 1):
            if self.tsp.distanceMatrix[self.order[i]][self.order[i + 1]] < minLength:
                minIndex = i
                minLength = self.tsp.distanceMatrix[self.order[i]][self.order[i + 1]]
        return minIndex

    def swapMutationUpdateInfo(self):
        points = random.sample(range(0, self.tsp.getLength()), 2)
        self.updateFitnessAndAdjacency(points)
        self.updateOrder(points)

    def updateOrder(self, points):
        self.order[points[0]], self.order[points[1]] = self.order[points[1]], self.order[points[0]]

    def updateFitnessAndAdjacency(self, points):
        smallerPoint = min(points[0], points[1])
        biggerPoint = max(points[0], points[1])
        if abs(points[0] - points[1]) == 1:
            p1 = self.order[smallerPoint - 1]
            p2 = self.order[smallerPoint]
            p3 = self.order[smallerPoint + 1]
            p4 = self.order[smallerPoint + 2] if smallerPoint + 2 != self.tsp.length else self.order[0]
            d1 = self.getAdaptedDistance(self.tsp.distanceMatrix[p1][p2])
            d2 = self.getAdaptedDistance(self.tsp.distanceMatrix[p2][p3])
            d3 = self.getAdaptedDistance(self.tsp.distanceMatrix[p3][p4])
            newD1 = self.getAdaptedDistance(self.tsp.distanceMatrix[p1][p3])
            newD2 = self.getAdaptedDistance(self.tsp.distanceMatrix[p3][p2])
            newD3 = self.getAdaptedDistance(self.tsp.distanceMatrix[p2][p4])
            self.fitnessValue += newD1 + newD2 + newD3 - d1 - d2 - d3
            if abs(self.fitnessValue) == math.inf:
                print("Problem occurs")
            self.adjacency[p1] = p3
            self.adjacency[p3] = p2
            self.adjacency[p2] = p4
        elif abs(points[0] - points[1]) == self.tsp.length - 1:
            p1 = self.order[self.tsp.length - 2]
            p2 = self.order[self.tsp.length - 1]
            p3 = self.order[0]
            p4 = self.order[1]
            d1 = self.getAdaptedDistance(self.tsp.distanceMatrix[p1][p2])
            d2 = self.getAdaptedDistance(self.tsp.distanceMatrix[p2][p3])
            d3 = self.getAdaptedDistance(self.tsp.distanceMatrix[p3][p4])
            newD1 = self.getAdaptedDistance(self.tsp.distanceMatrix[p1][p3])
            newD2 = self.getAdaptedDistance(self.tsp.distanceMatrix[p3][p2])
            newD3 = self.getAdaptedDistance(self.tsp.distanceMatrix[p2][p4])
            self.fitnessValue += newD1 + newD2 + newD3 - d1 - d2 - d3
            if abs(self.fitnessValue) == math.inf:
                print("Problem occurs")
            self.adjacency[p1] = p3
            self.adjacency[p3] = p2
            self.adjacency[p2] = p4
        else:
            p1 = self.order[smallerPoint - 1]
            p2 = self.order[smallerPoint]
            p3 = self.order[smallerPoint + 1]
            p4 = self.order[biggerPoint - 1]
            p5 = self.order[biggerPoint]
            p6 = self.order[biggerPoint + 1] if biggerPoint + 1 != self.tsp.length else self.order[0]
            d1 = self.getAdaptedDistance(self.tsp.distanceMatrix[p1][p2])
            d2 = self.getAdaptedDistance(self.tsp.distanceMatrix[p2][p3])
            d3 = self.getAdaptedDistance(self.tsp.distanceMatrix[p4][p5])
            d4 = self.getAdaptedDistance(self.tsp.distanceMatrix[p5][p6])
            newD1 = self.getAdaptedDistance(self.tsp.distanceMatrix[p1][p5])
            newD2 = self.getAdaptedDistance(self.tsp.distanceMatrix[p5][p3])
            newD3 = self.getAdaptedDistance(self.tsp.distanceMatrix[p4][p2])
            newD4 = self.getAdaptedDistance(self.tsp.distanceMatrix[p2][p6])
            self.fitnessValue += newD1 + newD2 + newD3 + newD4 - d1 - d2 - d3 - d4
            if abs(self.fitnessValue) == math.inf:
                print("Problem occurs")
            self.adjacency[p1] = p5
            self.adjacency[p5] = p3
            self.adjacency[p4] = p2
            self.adjacency[p2] = p6

    def getAdaptedDistance(self, distance):
        return distance if distance != math.inf else self.maxValue * (self.penalty ** self.penaltyOrder)


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
        self.bestSolution = None
        self.islandPopulation = None
        self.islandOffspring = []
        self.islandSize = 0

    def initiatePopulation(self):
        for times in range(self.populationSize):
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
            self.population = np.append(self.population, [newIndividual])
        print("Initialization end")

    def createOffsprings(self):
        self.offspringPopulation = []
        for i in range(self.nbOffsprings):
            if random.random() < self.recombineProbability:
                o1, o2 = self.partiallyMappedCrossoverRecombination()
                self.offspringPopulation = np.append(self.offspringPopulation, [o1])
                self.offspringPopulation = np.append(self.offspringPopulation, [o2])
        self.population = np.append(self.population, [self.offspringPopulation])

    def partiallyMappedCrossoverRecombination(self, mode=0):
        p1 = p2 = 0
        if mode == 0:
            p1, p2 = self.sharedSelection()
        else:
            players = list(np.random.choice(self.population, size=self.tournament_k))
            players.sort(key=lambda p: p.fitnessValue)
            p1 = players[0]
            p2 = players[1]
        points = random.sample(range(0, self.tspLength - 1), 2)
        points.sort()
        if p1 == p2:
            print("Oh my God!")
        offspring1 = np.zeros(self.tspLength, dtype=int)
        offspring2 = np.zeros(self.tspLength, dtype=int)
        left = points[0]
        right = points[1]
        u1 = [-1 for i in range(self.tspLength)]
        u2 = [-1 for i in range(self.tspLength)]
        position1 = [-1 for i in range(self.tspLength)]
        position2 = [-1 for i in range(self.tspLength)]
        notUsedList1 = np.ones(self.tspLength, dtype=int)
        notUsedList2 = np.ones(self.tspLength, dtype=int)
        for i in range(self.tspLength):
            position1[p1.order[i]] = i
            position2[p2.order[i]] = i
        for i in range(left, right, 1):
            u1[p2.order[i]] = p1.order[i]
            u2[p1.order[i]] = p2.order[i]
            offspring1[i] = p1.order[i]
            offspring2[i] = p2.order[i]
            notUsedList1[p1.order[i]] = 0
            notUsedList2[p2.order[i]] = 0
        for i in range(left - self.tspLength, left, 1):
            cache = p2.order[i]
            if not notUsedList1[cache]:
                continue
            nextIndex = cache
            nextPos = u1[cache]
            while nextPos != -1 and nextPos != cache:
                nextIndex = nextPos
                nextPos = u1[nextPos]
            offspring1[position2[nextIndex]] = cache
        for i in range(left - self.tspLength, left, 1):
            cache = p1.order[i]
            if not notUsedList2[cache]:
                continue
            nextIndex = cache
            nextPos = u2[cache]
            while nextPos != -1 and nextPos != cache:
                nextIndex = nextPos
                nextPos = u2[nextPos]
            offspring2[position1[nextIndex]] = cache
        o1 = Individual(self.tsp, self.penalty, self.maxValue, offspring1)
        o2 = Individual(self.tsp, self.penalty, self.maxValue, offspring2)
        return o1, o2

    # order crossover operator
    def orderCrossoverOperator(self):
        players = list(np.random.choice(self.islandPopulation, size=2))
        p1 = players[0]
        p2 = players[1]
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
        o2 = Individual(self.tsp, self.penalty, self.maxValue, offspring2)
        return o1, o2

    def select(self):
        players = list(np.random.choice(self.population, size=self.tournament_k))
        return min(players, key=lambda p: p.fitnessValue)

    def selectTwoParents(self):
        p1 = self.select()
        p2 = self.select()
        t = 0
        while p1 == p2 and t < 5:
            p2 = self.select()
            t = t + 1
        return p1, p2

    def mutate(self):

        for i in self.population:
            if i == self.bestSolution:
                continue
            if random.random() < self.mutationProbability:
                # i.mutate()
                i.swapMutationUpdateInfo()

    def eliminate(self, iteration):

        # alpha = 0.99
        # s = alpha ** iteration
        # a = math.log(s)

        # s = 0.01
        # a = (1 - s) / ((300 - 1) ** 2)
        #
        # l = []
        # for i in range(len(self.population)):
        #     l.append(a * (i - 300)**2 + s)
        # probabilitySum = sum(l)
        #
        # p = list(self.population)
        # p.sort(key=lambda p: p.fitnessValue)
        # newPop = []
        # num = 0
        # while len(newPop) < self.populationSize:
        #     for i in range(len(p)):
        #         possibility = np.random.random() * 1.5
        #         if possibility < l[i]:
        #             num += 1
        #             newPop.append(p[i])
        #             if num >= self.populationSize:
        #                 break
        #
        # self.population = np.array(newPop)
        pop = list(self.population)
        pop.sort(key=lambda p: p.fitnessValue)
        topK = 5
        newPop = pop[:topK]
        for w in range(topK):
            for m, p in enumerate(self.population):
                if p == newPop[w]:
                    self.population = np.delete(self.population, m)
                    break

        for j in range(self.populationSize - topK):
            players = np.random.choice(self.population, size=self.tournament_k)
            minFitness = math.inf
            minIndex = 0
            for i, p in enumerate(players):
                fitnessValue = self.sharedFitness(p, players)
                if fitnessValue < minFitness:
                    minFitness = fitnessValue
                    minIndex = i
            newPop.append(players[minIndex])

            for m, p in enumerate(self.population):
                if p == players[minIndex]:
                    self.population = np.delete(self.population, m)
                    break

        self.population = np.array(newPop)

    def getMeanFitness(self):
        return np.mean([i.fitnessValue for i in self.population])

    def getBestFitness(self):
        bestFitness = self.population[0].fitnessValue
        bestIndividual = self.population[0]
        totalFitness = 0
        for i in self.population:
            if i.fitnessValue < bestFitness:
                bestFitness = i.fitnessValue
                bestIndividual = i
            totalFitness += i.fitnessValue
        realAverage = totalFitness / len(self.population)
        self.bestSolution = bestIndividual
        return realAverage, bestFitness, bestIndividual

    def printInfiniteValueNums(self, individual):
        infNums = 0
        for i in range(self.tspLength - 1):
            if self.tsp.distanceMatrix[individual.order[i]][individual.order[i + 1]] == math.inf:
                infNums += 1
        if self.tsp.distanceMatrix[individual.order[self.tspLength - 1]][individual.order[0]] == math.inf:
            infNums += 1
        print("This new individual has fitness value: ", individual.fitnessValue)
        print("And it has ", infNums, " infinite values.")

    def getSimilarityOfTwoIndividuals(self, i1: Individual, i2: Individual):
        adjacencyList1 = i1.adjacency
        adjacencyList2 = i2.adjacency
        count = 0
        for i in range(self.tspLength):
            if adjacencyList1[i] == adjacencyList2[i]:
                count += 1
        return count / self.tspLength

    def sharedFitness(self, i1: Individual, pop=None):
        if pop is None:
            return i1.fitnessValue
        alpha = 0.5
        onePlusBeta = 1
        for i, p in enumerate(pop):
            if p == i1:
                continue
            similarity = self.getSimilarityOfTwoIndividuals(p, i1)
            if similarity > 0.1:
                onePlusBeta += 1 - similarity ** alpha

        return i1.fitnessValue * onePlusBeta

    def sharedSelection(self):
        t1 = time.time()
        players = np.random.choice(self.population, size=self.tournament_k)
        min1 = min2 = math.inf
        min1Index = min2Index = 0
        for i, p in enumerate(players):
            fitnessValue = self.sharedFitness(p, players)
            if fitnessValue < min1:
                min2 = min1
                min1 = fitnessValue
                min2Index = min1Index
                min1Index = i
            elif fitnessValue < min2:
                min2 = fitnessValue
                min2Index = i
        p1 = players[min1Index]
        p2 = players[min2Index]
        t2 = time.time()
        # print("Shared selection needs time: ", t2 - t1)
        return p1, p2

    def mutateGenerateNewIndividual(self, individual: Individual):
        order = individual.order.copy()
        points = random.sample(range(0, self.tsp.getLength()), 2)
        order[points[0]], order[points[1]] = order[points[1]], order[points[0]]
        newIndividual = Individual(self.tsp, self.penalty, self.maxValue, order)
        # print("new mutate individual fitness: ", newIndividual.fitnessValue)
        # self.population = np.append(self.population, [newIndividual])
        return newIndividual

    def getTopKIndividuals(self, topK=1):
        p = list(self.population)
        p.sort(key=lambda p: p.fitnessValue)
        unchangedPop = p[topK:]
        p = p[:topK]
        for i in range(len(p)):
            subGroup = []
            # print("p[", i, "] original fitness: ", p[i].fitnessValue)
            for j in range(self.tspLength // 10):
                temp = self.mutateGenerateNewIndividual(p[i])
                subGroup = np.append(subGroup, [temp])
            subGroup = list(subGroup)
            subGroup.sort(key=lambda sub: sub.fitnessValue)
            if subGroup[0].fitnessValue < p[i].fitnessValue:
                p[i].replaceOrder(subGroup[0].order, subGroup[0].fitnessValue)

            if sum(p[i].order) != self.tspLength * (self.tspLength - 1) / 2:
                print("Mutate Sum error")
        self.population = np.append(p, unchangedPop)

    def getPopulationSimilarityData(self):

        tss1 = time.time()
        temp = []
        temp1 = []
        for i in range(len(self.population)):
            p = self.population[i]
            tempScore = []
            for j in range(len(self.population)):
                if i == j:
                    continue
                nextP = self.population[j]
                similarity = self.getSimilarityOfTwoIndividuals(p, nextP)
                tempScore = np.append(tempScore, [similarity])
            count = 0
            count1 = 0
            for m in range(len(tempScore)):
                if tempScore[m] == 1:
                    count += 1
                if tempScore[m] >= 0.75:
                    count1 += 1
            temp = np.append(temp, [count])
            temp1 = np.append(temp1, [count1])
        filename = "Similarity.csv"
        delimiter = ','
        outFile = open(filename, "w")
        for i in range(len(temp)):
            outFile.write(str(temp[i]) + delimiter)
        outFile.write('\n')
        for i in range(len(temp)):
            outFile.write(str(temp1[i]) + delimiter)
        outFile.close()
        tss2 = time.time()
        print("Calculate 200 Individuals similarity needs time: ", tss2 - tss1)

    def deleteDuplicateIndividuals(self):
        newPop = []
        p = list(self.population)
        p.sort(key=lambda individual: individual.fitnessValue)
        newPop = np.append(newPop, p[0])
        for i in range(len(self.population)):
            hasSame = False
            for j in range(len(newPop)):
                if (self.population[i].adjacency == newPop[j].adjacency).all():
                    hasSame = True
                    break
            if not hasSame:
                newPop = np.append(newPop, [self.population[i]])

        self.population = newPop
        print("New population has ", len(self.population), " individuals")
        print("New population has best fitness ", newPop[0].fitnessValue)

    def getOneIsland(self, islandThreshold=0.75):
        print("Before pruned population size: ", len(self.population))
        l1 = list(self.population)
        bestIndividual1 = min(l1, key=lambda p: p.fitnessValue)
        print("bestIndividual1 has fitness ", bestIndividual1.fitnessValue)

        temp1 = []
        maxSimilarity = 0
        for i in range(len(self.population)):
            p = self.population[i]
            tempScore = []
            for j in range(len(self.population)):
                if i == j:
                    continue
                nextP = self.population[j]
                similarity = self.getSimilarityOfTwoIndividuals(p, nextP)
                maxSimilarity = max(maxSimilarity, similarity)
                tempScore = np.append(tempScore, [similarity])
            count1 = 0
            for m in range(len(tempScore)):
                if tempScore[m] >= islandThreshold:
                    count1 += 1
            temp1 = np.append(temp1, [count1])
        maxSameIndex = 0
        maxSameNumber = max(temp1)
        for i in range(len(temp1)):
            if temp1[i] == maxSameNumber:
                maxSameIndex = i
                break
        self.islandPopulation = []
        indexes = []
        for n in range(len(self.population)):
            if self.getSimilarityOfTwoIndividuals(self.population[maxSameIndex], self.population[n]) >= islandThreshold:
                self.islandPopulation = np.append(self.islandPopulation, [self.population[n]])
                indexes = np.append(indexes, [n])
        print(indexes)
        minusCount = 0
        for i in range(len(self.islandPopulation)):
            self.population = np.delete(self.population, int(indexes[i] - minusCount))
            minusCount += 1

        self.islandSize = len(self.islandPopulation)
        self.populationSize = len(self.population)
        print("Island population size: ", len(self.islandPopulation))
        print("After pruned population size: ", len(self.population))
        l1 = list(self.population)
        bestIndividual1 = min(l1, key=lambda p: p.fitnessValue)
        print("bestIndividual1 has fitness ", bestIndividual1.fitnessValue)
        l2 = list(self.islandPopulation)
        bestIndividual2 = min(l2, key=lambda p: p.fitnessValue)
        print("bestIndividual1 has fitness ", bestIndividual2.fitnessValue)

    def parallelCreatingOffspring(self):
        l1 = list(self.population)
        bestIndividual1 = min(l1, key=lambda p: p.fitnessValue)
        print("bestIndividual1 has fitness ", bestIndividual1.fitnessValue)
        self.offspringPopulation = []
        for i in range(len(self.population) // 2):
            if random.random() < self.recombineProbability:
                o1, o2 = self.partiallyMappedCrossoverRecombination(1)
                self.offspringPopulation = np.append(self.offspringPopulation, [o1, o2])
        self.population = np.append(self.population, self.offspringPopulation)

        l2 = list(self.islandPopulation)
        bestIndividual2 = min(l2, key=lambda p: p.fitnessValue)
        print("bestIndividual2 has fitness ", bestIndividual2.fitnessValue)
        self.offspringPopulation = []
        for i in range(len(self.islandPopulation) // 2):
            if random.random() < self.recombineProbability:
                o1, o2 = self.orderCrossoverOperator()
                self.offspringPopulation = np.append(self.offspringPopulation, [o1, o2])
        self.islandPopulation = np.append(self.islandPopulation, self.offspringPopulation)

    def parallelMutate(self):
        l1 = list(self.population)
        bestIndividual1 = min(l1, key=lambda p: p.fitnessValue)
        print("bestIndividual1 has fitness ", bestIndividual1.fitnessValue)
        for i in self.population:
            if i == bestIndividual1:
                continue
            if random.random() < self.mutationProbability * 5:
                i.swapMutationUpdateInfo()

        l2 = list(self.islandPopulation)
        bestIndividual2 = min(l2, key=lambda p: p.fitnessValue)
        print("bestIndividual2 has fitness ", bestIndividual2.fitnessValue)
        for j in self.islandPopulation:
            if j == bestIndividual2:
                continue
            if random.random() < self.mutationProbability * 5:
                j.swapMutationUpdateInfo()

    def parallelEliminate(self):
        p = list(self.population)
        p.sort(key=lambda individual: individual.fitnessValue)
        p = p[:self.populationSize]
        self.population = np.array(p)

        i = list(self.islandPopulation)
        i.sort(key=lambda individual: individual.fitnessValue)
        i = i[:self.islandSize]
        self.islandPopulation = np.array(i)

    def islandLocalSearch(self, topK=5):
        island = list(self.islandPopulation)
        island.sort(key=lambda p: p.fitnessValue)
        islandLeft = island[topK:]
        island = island[:topK]
        localSearchTime = self.tspLength // 10
        j = 0
        for individual in island:
            index = individual.findLongestPath()
            numList = [k for k in range(0, self.tspLength)]
            localSearch = random.sample(numList, localSearchTime)
            subGroup = [individual]
            for i in range(localSearchTime):
                order = individual.order.copy()
                order[index], order[localSearch[i]] = order[localSearch[i]], order[index]
                newIndividual = Individual(self.tsp, self.penalty, self.maxValue, order)
                subGroup = np.append(subGroup, [newIndividual])
            island[j] = min(subGroup, key=lambda p: p.fitnessValue)
            j += 1

        self.islandPopulation = np.append(island, islandLeft)

    def parallelLocalSearch(self):
        self.getTopKIndividuals(self.tournament_k)

        self.islandLocalSearch(self.tournament_k)

    def parallelGetBestFitness(self):
        bestFitness = self.population[0].fitnessValue
        bestIndividual = self.population[0]
        totalFitness = 0
        for i in self.population:
            totalFitness += i.fitnessValue
            if i.fitnessValue < bestFitness:
                bestFitness = i.fitnessValue
                bestIndividual = i
        for j in self.islandPopulation:
            totalFitness += j.fitnessValue
            if j.fitnessValue < bestFitness:
                bestFitness = j.fitnessValue
                bestIndividual = j
        avgFitness = totalFitness / (self.islandSize + self.populationSize)
        return avgFitness, bestFitness, bestIndividual

    def __call__(self, iteration, mode=0):
        if mode == 0:
            ts1 = time.time()
            self.createOffsprings()
            ts2 = time.time()
            print("Creating Offspring needs time: ", ts2 - ts1)
            self.mutate()
            ts3 = time.time()
            print("Mutation needs time: ", ts3 - ts2)
            self.getTopKIndividuals(5)
            ts4 = time.time()
            print("getTopKIndividuals needs time: ", ts4 - ts3)
            self.eliminate(iteration)
            ts5 = time.time()
            print("Elimination needs time: ", ts5 - ts4)
            mean, bestFitness, bestIndividual = self.getBestFitness()
            ts6 = time.time()
            print("getBestFitness needs time: ", ts6 - ts5)

            return mean, bestFitness, bestIndividual.order
        if mode == 1:
            # local search model
            self.parallelCreatingOffspring()
            self.parallelMutate()
            self.parallelLocalSearch()
            self.parallelEliminate()
            mean, bestFitness, bestIndividual = self.parallelGetBestFitness()
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
        maxValue = tsp.getMaxValue()

        algorithm = EvolutionaryAlgorithm(tsp, 200, 100, 0.1, 1, 5, maxValue)
        iterationTimes = 0

        # Your code here.
        yourConvergenceTestsHere = True
        # mode == 0 means using diversity selection and elimination method
        # mode == 1 means using local search
        mode = 0
        once = 0
        convergency = False
        convergentCount = 0
        while iterationTimes < 1000 and not convergency:
            meanObjective, bestObjective, bestSolution = algorithm(iterationTimes, mode)
            print(str(iterationTimes))
            print(str("{:10.2f}".format(meanObjective)))
            print(str("{:10.2f}".format(bestObjective)))
            print('\n')

            if meanObjective < bestObjective * 3 and iterationTimes >= 5 and once == 0:
                once = 1
                mode = 1
                algorithm.deleteDuplicateIndividuals()
                # algorithm.getPopulationSimilarityData()
                algorithm.getOneIsland()

            if mode == 1 and meanObjective < bestObjective * 1.15:
                convergentCount += 1
            else:
                convergentCount = 0

            if convergentCount >= 100:
                convergency = True
            counts = 0
            ts1 = time.time()
            for i in range(-1, tsp.getLength() - 1, 1):
                if tsp.distanceMatrix[bestSolution[i]][bestSolution[i + 1]] == math.inf:
                    counts = counts + 1
            ts2 = time.time()
            print("best solution has ", counts, " infinite value, needs time: ", ts2 - ts1)
            # Your code here.

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0

            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            ts3 = time.time()
            print("Time left: ", timeLeft)
            print("Report needs time: ", ts3 - ts2)
            iterationTimes += 1
            if timeLeft < 0:
                break
        # Your code here.
        return 0


if __name__ == '__main__':
    r = r0776947()
    r.optimize("tour1000.csv")
