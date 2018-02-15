import math
import util
import checkpoint_algorithms
import random

import numpy
from scipy.spatial.distance import pdist, squareform

import deap.tools.emo
import pareto

name = 'SPEA2_mut'

def run(cache, tools, creator):
    "run the parameter estimation"
    random.seed()

    parameters = len(cache.MIN_VALUE)

    LAMBDA = parameters * cache.settings['population']
    MU = int(math.ceil(cache.settings['keep']*LAMBDA))
    if MU < 2:
        MU = 2

    pop = cache.toolbox.population(n=LAMBDA)

    if "seeds" in cache.settings:
        seed_pop = [cache.toolbox.individual_guess([f(v) for f, v in zip(cache.settings['transform'], sublist)]) for sublist in cache.settings['seeds']]
        pop.extend(seed_pop)

    totalGenerations = parameters * cache.settings['generations']

    hof = pareto.ParetoFront(similar=util.similar)
    meta_hof = pareto.ParetoFront(similar=util.similar)

    #return checkpoint_algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
    #    cxpb=settings['crossoverRate'], mutpb=settings['mutationRate'], ngen=totalGenerations, settings=settings, halloffame=hof, tools=tools)

    result = checkpoint_algorithms.eaMuPlusLambda(pop, cache.toolbox, mu=MU, lambda_=LAMBDA,
        cxpb=cache.settings['crossoverRate'], mutpb=cache.settings['mutationRate'], ngen=totalGenerations, settings=cache.settings, 
        halloffame=hof, tools=tools, cache=cache, meta_hof = meta_hof)

    return result

def setupDEAP(cache, fitness, map_function, creator, base, tools):
    "setup the DEAP variables"
    creator.create("FitnessMax", base.Fitness, weights=[1.0] * cache.numGoals)
    creator.create("Individual", list, typecode="d", fitness=creator.FitnessMax, strategy=None)

    cache.toolbox.register("individual", util.generateIndividual, creator.Individual,
        len(cache.MIN_VALUE), cache.MIN_VALUE, cache.MAX_VALUE, cache)
    cache.toolbox.register("population", tools.initRepeat, list, cache.toolbox.individual)

    cache.toolbox.register("individual_guess", util.initIndividual, creator.Individual, cache)

    cache.toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20.0, low=cache.MIN_VALUE, up=cache.MAX_VALUE)

    if cache.adaptive:
        cache.toolbox.register("mutate", tools.mutPolynomialBounded, eta=20.0, low=cache.MIN_VALUE, up=cache.MAX_VALUE, indpb=1.0/len(cache.MIN_VALUE))
        cache.toolbox.register("force_mutate", tools.mutPolynomialBounded, eta=20.0, low=cache.MIN_VALUE, up=cache.MAX_VALUE, indpb=1.0)
    else:
        cache.toolbox.register("mutate", tools.mutPolynomialBounded, eta=20.0, low=cache.MIN_VALUE, up=cache.MAX_VALUE, indpb=1.0/len(cache.MIN_VALUE))
        cache.toolbox.register("force_mutate", tools.mutPolynomialBounded, eta=20.0, low=cache.MIN_VALUE, up=cache.MAX_VALUE, indpb=1.0)

    cache.toolbox.register("select", selSPEA2)
    cache.toolbox.register("evaluate", fitness, json_path=cache.json_path)

    cache.toolbox.register('map', map_function)

#From DEAP with vectorization
######################################
# Strength Pareto         (SPEA-II)  #
######################################

def selSPEA2(individuals, k):
    """Apply SPEA-II selection operator on the *individuals*. Usually, the
    size of *individuals* will be larger than *n* because any individual
    present in *individuals* will appear in the returned list at most once.
    Having the size of *individuals* equals to *n* will have no effect other
    than sorting the population according to a strength Pareto scheme. The
    list returned contains references to the input *individuals*. For more
    details on the SPEA-II operator see [Zitzler2001]_.
    
    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :returns: A list of selected individuals.
    
    .. [Zitzler2001] Zitzler, Laumanns and Thiele, "SPEA 2: Improving the
       strength Pareto evolutionary algorithm", 2001.
    """
    N = len(individuals)
    L = len(individuals[0].fitness.values)
    K = math.sqrt(N)
    strength_fits = [0] * N
    fits = [0] * N
    dominating_inds = [list() for i in range(N)]
    
    for i, ind_i in enumerate(individuals):
        for j, ind_j in enumerate(individuals[i+1:], i+1):
            if ind_i.fitness.dominates(ind_j.fitness):
                strength_fits[i] += 1
                dominating_inds[j].append(i)
            elif ind_j.fitness.dominates(ind_i.fitness):
                strength_fits[j] += 1
                dominating_inds[i].append(j)
    
    for i in range(N):
        for j in dominating_inds[i]:
            fits[i] += strength_fits[j]
    
    # Choose all non-dominated individuals
    chosen_indices = [i for i in range(N) if fits[i] < 1]
    
    if len(chosen_indices) < k:     # The archive is too small
        for i in range(N):
            distances = [0.0] * N
            for j in range(i + 1, N):
                dist = 0.0
                for l in range(L):
                    val = individuals[i].fitness.values[l] - \
                          individuals[j].fitness.values[l]
                    dist += val * val
                distances[j] = dist
            kth_dist = deap.tools.emo._randomizedSelect(distances, 0, N - 1, K)
            density = 1.0 / (kth_dist + 2.0)
            fits[i] += density
            
        next_indices = [(fits[i], i) for i in range(N)
                        if not i in chosen_indices]
        next_indices.sort()
        #print next_indices
        chosen_indices += [i for _, i in next_indices[:k - len(chosen_indices)]]
                
    elif len(chosen_indices) > k:   # The archive is too large
        N = len(chosen_indices)
        distances = [[0.0] * N for i in range(N)]
        sorted_indices = [[0] * N for i in range(N)]
        
        vec_fitness = numpy.array([individuals[i].fitness.values for i in chosen_indices])
        vec_distances = squareform(pdist(vec_fitness, 'sqeuclidean'))
        numpy.fill_diagonal(vec_distances, -1)
        distances = vec_distances

        sorted_indices = numpy.argsort(distances, 1)

        to_remove = trim_individuals(k, N, distances, sorted_indices)

        #print("To remove = ", to_remove)
        for index in reversed(sorted(to_remove)):
            del chosen_indices[index]
    
    return [individuals[i] for i in chosen_indices]

def trim_individuals(k, N, distances, sorted_indices):
    size = N
    to_remove = []
    while size > k:
        # Search for minimal distance
        min_pos = 0
        for i in range(1, N):
            for j in range(1, size):
                dist_i_sorted_j = distances[i,sorted_indices[i,j]]
                dist_min_sorted_j = distances[min_pos,sorted_indices[min_pos,j]]

                if dist_i_sorted_j < dist_min_sorted_j:
                    min_pos = i
                    break
                elif dist_i_sorted_j > dist_min_sorted_j:
                    break
            
        distances[:,min_pos] = numpy.inf
        distances[min_pos,:] = numpy.inf

        #This part is still expensive but I don't know a better way to do it yet.
        #Essentially all remaining time in this function is in this section
        #It may even make sense to do this in C++ later since it is trivially parallel
        for i in range(N):
            for j in range(1, size - 1):
                if sorted_indices[i,j] == min_pos:
                    sorted_indices[i,j:size - 1] = sorted_indices[i,j + 1:size]
                    sorted_indices[i,j:size] = min_pos
                    break
            
        # Remove corresponding individual from chosen_indices
        to_remove.append(min_pos)
        size -= 1
    return to_remove
