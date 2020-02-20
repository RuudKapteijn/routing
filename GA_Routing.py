##############################################################################################################
# test script for routing library
#
##############################################################################################################
import Routing as rt
import numpy as np
from sys import version
from datetime import datetime
from random import randrange, random, uniform, randint


def ind_fitness(ind):
    # print(f"{ind.dist_to_go()}, {ind.elap_time()}")

    if ind.dist_to_go() < 0.01:             # destination reached
        distance_score = 10000
    else:
        distance_score = int(100 / ind.dist_to_go())

    time_score = int(6000000 / ind.elap_time())

    return distance_score + time_score * 5


def sort_population_by_fitness(population):
    return sorted(population, key=ind_fitness)


def choice_by_roulette(sorted_population, fitness_sum):
    offset = 0
    normalized_fitness_sum = fitness_sum

    lowest_fitness = ind_fitness(sorted_population[0])
    if lowest_fitness < 0:
        offset = -lowest_fitness
        normalized_fitness_sum += offset * len(sorted_population)

    draw = uniform(0, 1)

    accumulated = 0
    for individual in sorted_population:
        fitness = ind_fitness(individual) + offset
        probability = fitness / normalized_fitness_sum
        accumulated += probability

        if draw <= accumulated:
            return individual

def generate_genes():

    genes = []
    for i in range(0, 15):
        genes.append(randint(0, 359))
    return genes


def get_genes(individual):
    # Itinarary course list is considered the genes
    return individual.get_courselist()


def crossover_genes(genes1, genes2):
    """Crossover_genes make a new set of genes as a combination of two given sets

    :param genes1: 1st set of genes
    :param genes2: 2nd set of genes
    :return: list of int - new set of genes
    """
    new_genes = []
    i = 0
    while i < len(genes1) and i < len(genes2):
        if randint(0, 1) == 0:                    # random value 0 or 1
            new_genes.append(genes1[i])
        else:
            new_genes.append(genes2[i])
        i += 1
    while i < len(genes1):
      new_genes.append(genes1[i])
      i += 1
    while i < len(genes2):
      new_genes.append(genes2[i])
      i += 1
    if len(new_genes) != max(len(genes1), len(genes2)):
        print(f"crossover_genes() error - genes1: {str(genes1)}, genes2: {str(genes2)}, new_genes: {str(new_genes)}")
    return new_genes


def mutate_genes(genes):
    mutation_list = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    for i in range(0, 3):         # number of mutations
        mutation = mutation_list[randint(0, len(mutation_list) - 1)]
        index = randint(0, len(genes) - 1)
        # print(f"mutate_genes, index: {index}, len(genes): {len(genes)}, genes: {str(genes)}, mutation: {mutation}")
        genes[index] = (genes[index] + mutation) % 360
    if genes[index] < 0 or genes[index] > 359:
        print(f"mutate_genes() error – index: {index}, genes[index]: {genes[index]}")
    return genes


def generate_individual_with_genes(id, wind, polar, step, route, start_time, genes):
    ind = rt.Itinerary(id=id, route=route, start_time=start_time)  # create first itinerary/individual
    i = 0
    while i < len(genes) and ind.dist_to_go() > 0.1:  # add track points to individual with random direction
        ind.add_step(wind, polar, step, genes[i])
        i += 1
    return ind


def generate_population(wind, polar, route, start_time, step, size):
    population = []
    for i in range(0, size):
        genes = generate_genes()
        ind = generate_individual_with_genes(str(i), wind, polar, step, route, start_time, genes)
        population.append(ind)  # add individual to population
    return population


def make_next_generation(previous_population, wo, po, rte, start_time, step):
    next_generation = []
    sorted_by_fitness_population = sort_population_by_fitness(previous_population)
    population_size = len(previous_population)
    fitness_sum = sum(ind_fitness(individual) for individual in population)
    for i in range(population_size):
        first_choice = choice_by_roulette(sorted_by_fitness_population, fitness_sum)
        second_choice = choice_by_roulette(sorted_by_fitness_population, fitness_sum)
        genes1 = get_genes(first_choice)
        genes2 = get_genes(second_choice)
        crossed_genes = crossover_genes(genes1, genes2)
        mutated_genes = mutate_genes(crossed_genes)
        # new_id = first_choice.get_id()+"X"+second_choice.get_id()
        individual = generate_individual_with_genes("M", wo, po, step, rte, start_time, mutated_genes)
        next_generation.append(individual)

    # return next_generation
    blended_generation = sort_population_by_fitness(previous_population + next_generation)
    return blended_generation[population_size * -1:]


# start of script
print(f"Start of RoutingTest running on {version} at {datetime.now()}")

# Initialization
STEPTIME = 0.25  # steps of 15 minutes / quarter hour
mywo = rt.WindObject('gfs_new.grb.netcdf')  # open netcdf version of a GRIB file in object mywo
mypo = rt.PolarObject('Mango RaceV0.txt')  # open Expedition polar file in object mypo
org = rt.WayPoint("EA1", "52° 34,777'", "5° 13,433'")
dst = rt.WayPoint("KG10", "52° 40,656'", "5° 16,039'")
rte = rt.Route("myRoute")
rte.add_wpt(org)
rte.add_wpt(dst)
start_time = np.datetime64('2020-01-31 12:00')
print(f"{rte.stats()} at {str(start_time)}")

# generate population - set of itineraries
population = generate_population(mywo, mypo, rte, start_time, STEPTIME, 500)
print(f"\nInitial Population of {len(population)} individuals")

generations = 50
i = 1
while True:
    sorted_population = sort_population_by_fitness(population)
    best = sorted_population[-1]
    print("Generation %2d, population: %4d - best individual dist: %6.3f nm, time: %5.0f sec, fitness: %d, courses: %s" %
          (i, len(sorted_population), best.dist_to_go(), best.elap_time(), ind_fitness(best), str(best.get_courselist())))
    # for j in range(0, best.tpt_count()):
    #     print(f"Trackpoint {j}, time: {best.get_tpt(j).get_time()}")

    if i == generations:
        break
    population = make_next_generation(population, mywo, mypo, rte, start_time, STEPTIME)
    i += 1

print("\nOptimal result: dist: 0.00, minutes: 41.0, fitness: 22195, courses: [15, 15, 15]")

print(f"End of script at {datetime.now()}")
