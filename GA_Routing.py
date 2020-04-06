##############################################################################################################
# test script for routing library
#
##############################################################################################################

import Routing as rt
from sys import version
from datetime import datetime
from time import perf_counter
from random import random, uniform, randint, seed
import math
from math import degrees, floor, radians
# from numpy import array
# import matplotlib.pyplot as plt
from typing import List, Dict
import configparser as cp
import sys
import os

stats = {                         # dict to administer runtime statistics e.g.about errors
    'grib_index_exceptions': 0,
    'add_step'             : 0,
    'ind_fitness'          : 0
}

def ind_fitness(ind: rt.Itinerary) -> float:
    # print(f"{ind.dist_to_go()}, {ind.elap_time()}")
    stats['ind_fitness'] += 1

    if ind.dist_to_go() < 0.01:             # destination reached
        distance_score = 10000
    else:
        distance_score = int(100 / ind.dist_to_go())        # remaining distance is in nm

    if ind.elap_time() > 1:
        time_score = int(6000000 / ind.elap_time())  # elap_time is in seconds
    else:
        time_score = 6000

    speed_score = ind.avg_bsp()

    return distance_score + time_score * 5 + speed_score
# ind_fitness()


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

def generate_genes(genes_length: int, first_leg_direction: int):
    genes: List[float] = []
    beatlist = [60, 55, 50, 45, -45, -50, -55, -60]
    runlist  = [140, 150, 160, 170, -170, -160, -150, -140]

    scenario = randint(0, 3)

    if scenario <= 1:                                   # fully random
        for i in range(0, genes_length):
            genes.append(random() * 2 * math.pi)        # get random course 0 - 359 in radians

    if scenario == 2:                                   # beat
        for i in range(0, genes_length):
            j = randint(0, len(beatlist) - 1)
            c = (first_leg_direction + beatlist[j]) % 360
            genes.append(radians(c))

    if scenario == 3:                                   # run
        for i in range(0, genes_length):
            j = randint(0, len(runlist) - 1)
            c = (first_leg_direction + runlist[j]) % 360
            genes.append(radians(c))

    return genes

def get_genes(individual: rt.Itinerary) -> List[float]:         # Itinarary course list is considered the genes
    return individual.get_courselist()

def crossover_genes(genes1: List[float], genes2: List[float]) -> List[float]:
    """Crossover_genes make a new set of genes as a combination of two given sets

    :param genes1: 1st set of genes
    :param genes2: 2nd set of genes
    :return: new set of genes
    """
    new_genes: List[float] = []
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
    return new_genes
# crossover_genes()

def mutate_genes(genes: List[float], mutations: int, mutation_factor: float) -> List[float]:
    mutation_list = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    for i in range(0, mutations):         # number of mutations
        mutation = mutation_list[randint(0, len(mutation_list) - 1)] * 2 * math.pi / 360 * mutation_factor
        index = randint(0, len(genes) - 1)
        # print(f"mutate_genes, index: {index}, len(genes): {len(genes)}, genes: {str(genes)}, mutation: {mutation}")
        genes[index] = (genes[index] + mutation) % (2 * math.pi)
    return genes
# mutate_genes()

def generate_individual_with_genes(id, wind, polar, step, route, start_time, genes) -> rt.Itinerary:
    ind = rt.Itinerary(id=id, route=route, start_time=start_time)  # create first itinerary/individual
    i = 0
    while i < len(genes) and ind.dist_to_go() > 0.1:  # add track points to individual with random direction
        try:
            stats['add_step'] += 1
            ind.add_step(wind, polar, step, genes[i])
        except rt.GribIndexException as gie:
            if stats['grib_index_exceptions'] == 0:
                print(f"{gie.message} may indicate too big step or too many steps")
            stats['grib_index_exceptions'] += 1
        finally:
            i += 1
    return ind


def generate_population(wind, polar, route, start_time, step, size, genes_length, first_leg_direction, pop: List[rt.Itinerary] = None):
    # print("Generate population: ", end='')
    if pop:
        population = pop
    else:
        population = []
    l = len(population)

    for i in range(l, size):
        # print(f"{i}, ", end='')
        genes = generate_genes(genes_length, first_leg_direction)
        ind = generate_individual_with_genes(str(i), wind, polar, step, route, start_time, genes)
        population.append(ind)  # add individual to population
    # print("")
    return population


def make_next_generation(previous_population, wo, po, rte, start_time, step, mutations, mutation_factor):
    next_generation = []
    sorted_by_fitness_population = sorted(previous_population, key=ind_fitness)
    population_size = len(previous_population)
    fitness_sum = sum(ind_fitness(individual) for individual in population)
    for i in range(population_size):
        first_choice = choice_by_roulette(sorted_by_fitness_population, fitness_sum)
        second_choice = choice_by_roulette(sorted_by_fitness_population, fitness_sum)
        genes1 = get_genes(first_choice)
        genes2 = get_genes(second_choice)
        crossed_genes = crossover_genes(genes1, genes2)
        mutated_genes = mutate_genes(crossed_genes, mutations, mutation_factor)
        # new_id = first_choice.get_id()+"X"+second_choice.get_id()
        individual = generate_individual_with_genes("M", wo, po, step, rte, start_time, mutated_genes)
        next_generation.append(individual)

    # return next_generation
    blended_generation = sorted(previous_population + next_generation, key=ind_fitness)
    return blended_generation[population_size * -1:]


def get_parameters(file='config.ini'):
    params = cp.ConfigParser()
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        params.read(sys.argv[1], encoding='utf-8')
        print(f"Configuration parameters coming from {sys.argv[1]}")
        return params
    if os.path.isfile(file):
        params.read(file, encoding='utf-8')
        print(f"Configuration parameters coming from {file}")
        return params
    print("No configuration parameters found")
# get_parameters()

def coord_string(c: float) -> str:
    deg = floor(c)
    min = (c - deg) * 60
    return f"{deg:003d} {min:006.3f}"

def print_winning_track(ind: rt.Itinerary) -> None:
    print("\nSp Date       Time      Twd    Tws  Hea   Twa   Bsp   Dtg  Lat         Long")
    crslist = ind.get_courselist()
    dest: rt.WayPoint = ind.get_dest()
    for i in range(len(crslist)):
        tpt = ind.get_tpt(i)
        ts = str(tpt.get_time())[0:19]
        wind: Dict[str, float] = mywo.get_wind(tpt=tpt)
        twd: float = wind['twd']                        # radians
        tws: float = wind['tws']
        hea: float = crslist[i]                         # radians
        twa: float = mywo.TWA(twd=twd, hea=hea)         # degrees
        bsp: float = mypo.get_bsp(tws=tws, twa=twa)
        dtg: float = tpt.get_wpt().dtw(dest)
        lat: str = coord_string(tpt.get_lat())
        long: str = coord_string(tpt.get_long())
        print(f"{i:2d} {ts}  {degrees(twd):3.0f}  {tws:2.2f}  {degrees(hea):3.0f}  {twa:4.0f}  {bsp:2.2f}  {dtg:5.3f}  {lat}  {long}")
    tpt = ind.last_tpt()
    ts = str(tpt.get_time())[0:19]
    dtg: float = tpt.get_wpt().dtw(dest)
    lat: str = coord_string(tpt.get_lat())
    long: str = coord_string(tpt.get_long())
    print(f"-1 {ts}                               {dtg:5.3f}  {lat}  {long}\n")

def write_result_file(ind: rt.Itinerary) -> None:
    f_out = open('result.gpx', 'w')

    f_in  = open('header.xml', 'r')
    for line in f_in.readlines():
        f_out.write(line)
    f_in.close()

    suffix: List[str] = []
    f_in = open('suffix.xml', 'r')
    for line in f_in.readlines():
        suffix.append(line)
    f_in.close()

    crslist = ind.get_courselist()
    for i in range(len(crslist) + 1):
        tpt = ind.get_tpt(i)
        lat: str = str(tpt.get_lat())
        long: str = str(tpt.get_long())
        f_out.write(f'\t\t<rtept lat="{lat}" lon="{long}">\n')
        f_out.write("\t\t  <time>2020-03-29T16:05:19Z</time>\n")
        f_out.write(f'\t\t  <name>wpt{i}</name>\n')
        for line in suffix:
            f_out.write(line)

    f_out.write("\t\t</rte>\n")
    f_out.write("\t</gpx>\n")
    f_out.close()

# start of script
pf_step0 = perf_counter()
print(f"Start of GA_routing.py running on {version} at {datetime.now()}")
print(f"Uses {rt.version()}")

# Initialization
params = get_parameters()
STEPTIME = params['default'].getfloat('step_size')
par_seed = params['default'].getint('random_seed')
if par_seed and par_seed > 0:
    seed(par_seed)                   # set random seed to reproduce results
    print(f"Random seed set to {par_seed}")
# open netcdf version of a wind GRIB file and current GRIB file in object mywo
mywo = rt.WindObject(windfile=params['default']['wind_grib_file'], currentfile=params['default']['current_grib_file'])
mypo = rt.PolarObject(filename=params['default']['polar_file'])     # open Expedition polar file in object mypo
rte = rt.Route(name='-', gpxfile=params['default']['route_file'])   # open route from gpx file
fld = rte.wpt(0).btw(rte.wpt(1))
start_time: datetime = datetime.strptime(params['default']['start_time'], "%Y-%m-%d %H:%M")
generations = params['default'].getint('generations')
print(f"{rte.stats()} at start time: {str(start_time)}, generations: {generations}")

# generate initial population - set of itineraries
pf_step1 = perf_counter()
print(f"Initialization: {((pf_step1 - pf_step0) * 1000):.2f} ms")
population = generate_population(mywo, mypo, rte, start_time, STEPTIME,\
                                 size=params['default'].getint('inital_population_size'),\
                                 genes_length=params['default'].getint('genes_length'), first_leg_direction=fld)

i = 1
while True:
    j = 0
    while j < len(population) and len(population)> 1:
        if population[j].avg_bsp() < -1:
            del population[j]
        else:
            j += 1

    sorted_population = sorted(population, key=ind_fitness)
    best: rt.Itinerary = sorted_population[-1]
    hrs  = (best.elap_time() / 60) // 60        # floor division
    mins = (best.elap_time() / 60) % 60
    cl: List[int] = []
    for course in best.get_courselist():
        cl.append(round(math.degrees(course)))
    pf_step2 = perf_counter()
    print("Generation %2d (%5.0f s), population: %4d - best: %s dist: %6.3f nm, time: %d hr %6.3f min, avg bsp: %5.2f, fitness: %d, courses: %s" %
         (i, (pf_step2 - pf_step1), len(sorted_population), best.get_id(), best.dist_to_go(), hrs, mins, best.avg_bsp(), ind_fitness(best), str(cl)))

    # bsp_list: List[float] = []
    # for i in range(len(sorted_population)):
    #     ind: rt.Itinerary = sorted_population[i]
    #     bsp_list.append(ind.avg_bsp())
    # a = array(bsp_list)
    # plt.hist(a.astype('float'))
    # plt.show()

    if i == generations:
        break
    pf_step1 = perf_counter()
    population = make_next_generation(population, mywo, mypo, rte, start_time, STEPTIME, params['default'].getint('mutations'), params['default'].getint('mutation_factor'))
    population = generate_population(mywo, mypo, rte, start_time, STEPTIME,\
                                 size=params['default'].getint('inital_population_size'),\
                                 genes_length=params['default'].getint('genes_length'), first_leg_direction=fld,\
                                 pop=population)
    i += 1

write_result_file(best)
print_winning_track(best)
print(f"Number of Grib Index Exceptions: {stats['grib_index_exceptions']} (may indicate too large step or too many steps)")
print(f"Function calls ind_fitness: {stats['ind_fitness']}, add_step: {stats['ind_fitness']}")
print(f"End of script at {datetime.now()}")
