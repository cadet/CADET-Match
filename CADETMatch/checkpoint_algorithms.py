from pathlib import Path
import pickle
import random
import numpy
import util
from deap import algorithms
import gradFD
import time
import csv

def eaMuCommaLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, settings,
                    stats=None, halloffame=None, verbose=__debug__, tools=None, cache=None, meta_hof=None):
    """from DEAP function but with checkpoiting"""

    assert lambda_ >= mu, "lambda must be greater or equal to mu."

    checkpointFile = Path(settings['resultsDirMisc'], settings['checkpointFile'])

    sim_start = generation_start = time.time()

    path = Path(cache.settings['resultsDirBase'], cache.settings['CSV'])
    with path.open('a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)

        if checkpointFile.exists():
            with checkpointFile.open('rb') as cp_file:
                cp = pickle.load(cp_file)
            population = cp["population"]
            start_gen = cp["generation"]    
    
            halloffame = cp["halloffame"]
            meta_hof = cp['meta_halloffame']
            logbook = cp["logbook"]
            random.setstate(cp["rndstate"])
            gradCheck = cp['gradCheck']

        else:
            # Start a new evolution
            start_gen = 0    

            logbook = tools.Logbook()
            gradCheck = settings['gradCheck']


            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            util.eval_population(toolbox, cache, invalid_ind, writer, csvfile, halloffame, meta_hof)

            avg, bestMin, bestProd = util.averageFitness(population)
            util.writeProgress(cache, -1, population, halloffame, meta_hof, avg, bestMin, bestProd, sim_start, generation_start)
            util.graph_process(cache)

            logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=0, nevals=len(invalid_ind), **record)

            cp = dict(population=population, generation=start_gen, halloffame=halloffame,
                logbook=logbook, rndstate=random.getstate(), gradCheck=gradCheck, meta_halloffame=meta_hof)

            with checkpointFile.open('wb')as cp_file:
                pickle.dump(cp, cp_file)

        # Begin the generational process
        for gen in range(start_gen, ngen+1):
            generation_start = time.time()
            # Vary the population
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            offspring = util.RoundOffspring(cache, offspring, halloffame)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            util.eval_population(toolbox, cache, invalid_ind, writer, csvfile, halloffame, meta_hof)

            gradCheck, newChildren = gradFD.search(gradCheck, offspring, cache)
            offspring.extend(newChildren)

            avg, bestMin, bestProd = util.averageFitness(offspring)
            util.writeProgress(cache, gen, offspring, halloffame, meta_hof, avg, bestMin, bestProd, sim_start, generation_start)
            util.graph_process(cache)

            # Select the next generation population
            population[:] = toolbox.select(offspring, mu)
                   
            # Update the statistics with the new population
            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)

            cp = dict(population=population, generation=gen, halloffame=halloffame,
                logbook=logbook, rndstate=random.getstate(), gradCheck=gradCheck, meta_halloffame=meta_hof)

            hof = Path(settings['resultsDirMisc'], 'hof')
            with hof.open('wb') as data:
                numpy.savetxt(data, numpy.array(halloffame))
            with checkpointFile.open('wb') as cp_file:
                pickle.dump(cp, cp_file)

            if avg > settings['stopAverage'] or bestMin > settings['stopBest']:
                util.finish(cache)
                return halloffame
        util.finish(cache)
        return halloffame


def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, settings,
                   stats=None, halloffame=None, verbose=__debug__, tools=None, cache=None, meta_hof=None):
    """from DEAP function but with checkpoiting"""
    assert lambda_ >= mu, "lambda must be greater or equal to mu."

    #import search.spea2
    #from line_profiler import LineProfiler
    #profile = LineProfiler(search.spea2.selSPEA2)

    checkpointFile = Path(settings['resultsDirMisc'], settings['checkpointFile'])

    sim_start = generation_start = time.time()

    path = Path(cache.settings['resultsDirBase'], cache.settings['CSV'])
    with path.open('a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)

        if checkpointFile.exists():
            with checkpointFile.open('rb') as cp_file:
                cp = pickle.load(cp_file)
            population = cp["population"]
            start_gen = cp["generation"]    
    
            halloffame = cp["halloffame"]
            meta_hof = cp['meta_halloffame']
            logbook = cp["logbook"]
            random.setstate(cp["rndstate"])
            gradCheck = cp['gradCheck']
        else:
            # Start a new evolution
            start_gen = 0    

            logbook = tools.Logbook()
            gradCheck = settings['gradCheck']


            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            util.eval_population(toolbox, cache, invalid_ind, writer, csvfile, halloffame, meta_hof)

            avg, bestMin, bestProd = util.averageFitness(population)
            util.writeProgress(cache, -1, population, halloffame, meta_hof, avg, bestMin, bestProd, sim_start, generation_start)
            util.graph_process(cache)

            logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=0, nevals=len(invalid_ind), **record)

            cp = dict(population=population, generation=start_gen, halloffame=halloffame,
                logbook=logbook, rndstate=random.getstate(), gradCheck=gradCheck, meta_halloffame=meta_hof)

            with checkpointFile.open('wb')as cp_file:
                pickle.dump(cp, cp_file)

        # Begin the generational process
        for gen in range(start_gen, ngen+1):

            generation_start = time.time()
            # Vary the population
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            offspring = util.RoundOffspring(cache, offspring, halloffame)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            util.eval_population(toolbox, cache, invalid_ind, writer, csvfile, halloffame, meta_hof)

            # Combination of varOr and RoundOffSpring invalidates some members of the population, not sure why yet
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            util.eval_population(toolbox, cache, invalid_ind, writer, csvfile, halloffame, meta_hof)

            gradCheck, newChildren = gradFD.search(gradCheck, offspring, cache)
            offspring.extend(newChildren)

            avg, bestMin, bestProd = util.averageFitness(offspring)
            util.writeProgress(cache, gen, offspring, halloffame, meta_hof, avg, bestMin, bestProd, sim_start, generation_start)
            util.graph_process(cache)

            # Select the next generation population
            population[:] = toolbox.select(offspring + population, mu)
        
            # Update the statistics with the new population
            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)

            cp = dict(population=population, generation=gen, halloffame=halloffame,
                logbook=logbook, rndstate=random.getstate(), gradCheck=gradCheck, meta_halloffame=meta_hof)

            hof = Path(settings['resultsDirMisc'], 'hof')
            with hof.open('wb') as data:
                numpy.savetxt(data, numpy.array(halloffame))
            with checkpointFile.open('wb') as cp_file:
                pickle.dump(cp, cp_file)

            if avg > settings['stopAverage'] or bestMin > settings['stopBest']:
                util.finish(cache)
                return halloffame
        util.finish(cache)
        return halloffame

def varAnd(population, toolbox, cxpb, mutpb):
    """This is copied from the DEAP version but the mutation and crossover order are switched.
    This allows adaptive mutation to be used before the fitness scores are invalided.
    """
    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    return offspring
