import random
import math
import pickle
import util
import numpy
import array
from pathlib import Path
#import grad
import gradFD

from deap import algorithms

def run(cache, tools, creator):
    "run the parameter estimation"
    random.seed()

    parameters = len(cache.MIN_VALUE)

    populationSize=parameters * cache.settings['population']
    CXPB=cache.settings['crossoverRate']

    totalGenerations = parameters * cache.settings['generations']

    checkpointFile = Path(cache.settings['resultsDirMisc'], cache.settings['checkpointFile'])

    if checkpointFile.exists():
        with checkpointFile.open('rb') as cp_file:
            cp = pickle.load(cp_file)
        population = cp["population"]
        start_gen = cp["generation"]    
    
        halloffame = cp["halloffame"]
        logbook = cp["logbook"]
        random.setstate(cp["rndstate"])
        gradCheck = cp['gradCheck']

    else:
        # Start a new evolution

        population = cache.toolbox.population(n=populationSize)

        if "seeds" in cache.settings:
            seed_pop = [cache.toolbox.individual_guess([f(v) for f, v in zip(cache.settings['transform'], sublist)]) for sublist in cache.settings['seeds']]
            population.extend(seed_pop)

        start_gen = 0    

        halloffame = tools.ParetoFront(similar=util.similar)
        logbook = tools.Logbook()
        gradCheck = cache.settings['gradCheck']

        logbook.header = ['gen', 'nevals']
   
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = cache.toolbox.map(cache.toolbox.evaluate, map(list, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit    

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    population = cache.toolbox.select(population, len(population)) 
    
    avg, bestMin, bestProd = util.averageFitness(population)
    print('avg', avg, 'bestMin', bestMin, 'bestProd', bestProd)

    logbook.record(gen=start_gen, evals=len(invalid_ind))

    cp = dict(population=population, generation=start_gen, halloffame=halloffame,
        logbook=logbook, rndstate=random.getstate(), gradCheck=gradCheck)
    #cp = dict(population=population, generation=start_gen, halloffame=halloffame,
    #    logbook=logbook, rndstate=random.getstate())

    with checkpointFile.open('wb')as cp_file:
        pickle.dump(cp, cp_file)

    util.space_plots(cache)

    # Begin the generational process
    for gen in range(start_gen, totalGenerations+1):
        # Vary the population
        offspring = tools.selTournamentDCD(population, len(population))
        offspring = [cache.toolbox.clone(ind) for ind in offspring]
        
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                cache.toolbox.mate(ind1, ind2)
            
            cache.toolbox.mutate(ind1)
            cache.toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = cache.toolbox.map(cache.toolbox.evaluate, map(list, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("About to start gradient search")
        gradCheck, newChildren = gradFD.search(gradCheck, offspring, cache)
        print("Finished gradient search with new children", len(newChildren))
        offspring.extend(newChildren)

        avg, bestMin, bestProd = util.averageFitness(offspring)
        print('avg', avg, 'bestMin', bestMin, 'bestProd', bestProd)

        # Select the next generation population
        population = cache.toolbox.select(population + offspring, populationSize)
        logbook.record(gen=gen, evals=len(invalid_ind))

        #cp = dict(population=population, generation=gen, halloffame=halloffame,
        #    logbook=logbook, rndstate=random.getstate())

        cp = dict(population=population, generation=start_gen, halloffame=halloffame,
            logbook=logbook, rndstate=random.getstate(), gradCheck=gradCheck)

        hof = Path(cache.settings['resultsDirMisc'], 'hof')
        with hof.open('wb') as data:
            numpy.savetxt(data, numpy.array(halloffame))
        with checkpointFile.open('wb') as cp_file:
            pickle.dump(cp, cp_file)

        util.space_plots(cache)

        if avg > cache.settings['stopAverage'] or bestMin > cache.settings['stopBest']:
            return halloffame
    return halloffame

def setupDEAP(cache, fitness, map_function, creator, base, tools):
    "setup the DEAP variables"
    creator.create("FitnessMax", base.Fitness, weights=[1.0] * cache.numGoals)
    creator.create("Individual", list, typecode="d", fitness=creator.FitnessMax, strategy=None)

    cache.toolbox.register("individual", util.generateIndividual, creator.Individual,
        len(cache.MIN_VALUE), cache.MIN_VALUE, cache.MAX_VALUE)
    cache.toolbox.register("population", tools.initRepeat, list, cache.toolbox.individual)

    cache.toolbox.register("individual_guess", util.initIndividual, creator.Individual)

    cache.toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20.0, low=cache.MIN_VALUE, up=cache.MAX_VALUE)

    if cache.adaptive:
        cache.toolbox.register("mutate", util.mutPolynomialBoundedAdaptive, eta=10.0, low=cache.MIN_VALUE, up=cache.MAX_VALUE, indpb=1.0/len(cache.MIN_VALUE))
    else:
        cache.toolbox.register("mutate", tools.mutPolynomialBounded, eta=20.0, low=cache.MIN_VALUE, up=cache.MAX_VALUE, indpb=1.0/len(cache.MIN_VALUE))

    cache.toolbox.register("select", tools.selNSGA2)
    cache.toolbox.register("evaluate", fitness, json_path=cache.json_path)

    cache.toolbox.register('map', map_function)