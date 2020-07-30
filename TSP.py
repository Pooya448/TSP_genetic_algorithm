import numpy as np
import random, operator
import pandas as pd
import matplotlib.pyplot as plt

class city(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def compute_distance(self, other_city):
        return np.sqrt((abs(self.x - other_city.x) ** 2) + (abs(self.y - other_city.y) ** 2))

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0

    def compute_route_distance(self):
        if self.distance == 0:
            route_distance = 0
            for i in range(0, len(self.route)):
                from_city = self.route[i]
                to_city = None
                if i + 1 < len(self.route):
                    to_city = self.route[i + 1]
                else:
                    to_city = self.route[0]
                route_distance += from_city.compute_distance(to_city)
            self.distance = route_distance
        return self.distance

    def compute_fitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.compute_route_distance())
        return self.fitness

def create_route(city_list):
    route = random.sample(city_list, len(city_list))
    return route


def initial_population(pop_size, city_list):
    population = []
    for i in range(0, pop_size):
        population.append(create_route(city_list))
    return population


def rank_by_fitness(population):
    fitness_results = {}
    for i in range(0,len(population)):
        fitness_results[i] = Fitness(population[i]).compute_fitness()
    return sorted(fitness_results.items(), key = operator.itemgetter(1), reverse = True)

def selection(pop_rank, elite_size):
    selection_results = []
    df = pd.DataFrame(np.array(pop_rank), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()

    for i in range(0, elite_size):
        selection_results.append(pop_rank[i][0])
    for i in range(0, len(pop_rank) - elite_size):
        pick_prob = 100 * random.random()
        for i in range(0, len(pop_rank)):
            if pick_prob <= df.iat[i,3]:
                selection_results.append(pop_rank[i][0])
                break
    return selection_results

def create_mating_pool(population, selection_results):
    mating_pool = []
    for i in range(0, len(selection_results)):
        index = selection_results[i]
        mating_pool.append(population[index])
    return mating_pool



def breed(parent1, parent2):
    child = []
    child_part1 = []
    child_part2 = []

    gene_a = np.random.randint(0,len(parent1))
    gene_b = np.random.randint(0,len(parent1))

    start_gene = min(gene_a, gene_b)
    end_gene = max(gene_a, gene_b)

    for i in range(start_gene, end_gene):
        child_part1.append(parent1[i])

    child_part2 = [item for item in parent2 if item not in child_part1]

    child = child_part1 + child_part2
    print(len(child))
    return child


def breed_population(mating_pool, elite_size):
    children = []
    length = len(mating_pool) - elite_size
    pool = random.sample(mating_pool, len(mating_pool))

    for i in range(0,elite_size):
        children.append(mating_pool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(mating_pool) - i - 1])
        children.append(child)
    return children



def mutate(individual, mutation_rate):
    for swap1 in range(len(individual)):
        if(random.random() < mutation_rate):
            swap2 = np.random.randint(0,len(individual))

            city1 = individual[swap1]
            city2 = individual[swap2]

            individual[swap1] = city2
            individual[swap2] = city1
    return individual


def mutate_population(population, mutation_rate):
    mutated_population = []

    for index in range(0, len(population)):
        mutated_individual = mutate(population[index], mutation_rate)
        mutated_population.append(mutated_individual)
    return mutated_population

def next_generation(current_gen, elite_size, mutation_rate):
    elite_fitness = rank_by_fitness(current_gen)
    selection_results = selection(elite_fitness, elite_size)
    mating_pool = create_mating_pool(current_gen, selection_results)
    children = breed_population(mating_pool, elite_size)
    next_gen = mutate_population(children, mutation_rate)
    return next_gen

def GA(population, pop_size, elite_size, mutation_rate, no_of_generations):
    pop = initial_population(pop_size, population)
    print("Initial distance: " + str(1 / rank_by_fitness(pop)[0][1]))

    progress = []
    progress.append(1 / rank_by_fitness(pop)[0][1])

    for i in range(0, no_of_generations):
        pop = next_generation(pop, elite_size, mutation_rate)
        progress.append(1 / rank_by_fitness(pop)[0][1])

    print("Final distance: " + str(1 / rank_by_fitness(pop)[0][1]))

    best_route_index = rank_by_fitness(pop)[0][0]
    best_route = pop[best_route_index]
    # print("Best route: " + str(best_route))

    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()

    return best_route



city_list = []
for i in range(0,30):
    city_list.append(city(x=int(random.random() * 200), y=int(random.random() * 200)))

GA(population=city_list, pop_size=100, elite_size=20, mutation_rate=0.01, no_of_generations=500)
