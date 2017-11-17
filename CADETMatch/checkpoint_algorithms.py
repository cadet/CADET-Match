from pathlib import Path
import pickle
import random
import numpy
import util
from deap import algorithms
import gradFD

def eaMuCommaLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, settings,
                    stats=None, halloffame=None, verbose=__debug__, tools=None, cache=None):
    """from DEAP function but with checkpoiting"""

    assert lambda_ >= mu, "lambda must be greater or equal to mu."

    checkpointFile = Path(settings['resultsDirMisc'], settings['checkpointFile'])

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
        start_gen = 0    

        logbook = tools.Logbook()
        gradCheck = settings['gradCheck']


        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, map(list, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        avg, bestMin, bestProd = util.averageFitness(population)
        print('avg', avg, 'bestMin', bestMin, 'bestProd', bestProd)

        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        cp = dict(population=population, generation=start_gen, halloffame=halloffame,
            logbook=logbook, rndstate=random.getstate(), gradCheck=gradCheck)

        print('hof', halloffame)

        with checkpointFile.open('wb')as cp_file:
            pickle.dump(cp, cp_file)

    # Begin the generational process
    for gen in range(start_gen, ngen+1):
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, map(list, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("About to start gradient search")
        gradCheck, newChildren = gradFD.search(gradCheck, offspring, cache)
        print("Finished gradient search with new children", len(newChildren))
        offspring.extend(newChildren)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(offspring, mu)

        avg, bestMin, bestProd = util.averageFitness(population)
        print('avg', avg, 'bestMin', bestMin, 'bestProd', bestProd)
        
        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        if verbose:
            print(logbook.stream)

        cp = dict(population=population, generation=gen, halloffame=halloffame,
            logbook=logbook, rndstate=random.getstate(), gradCheck=gradCheck)

        print('hof', halloffame)

        hof = Path(settings['resultsDirMisc'], 'hof')
        with hof.open('wb') as data:
            numpy.savetxt(data, numpy.array(halloffame))
        with checkpointFile.open('wb') as cp_file:
            pickle.dump(cp, cp_file)

        if avg > settings['stopAverage'] or bestMin > settings['stopBest']:
            return halloffame
    return halloffame


def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, settings,
                   stats=None, halloffame=None, verbose=__debug__, tools=None, cache=None):
    """from DEAP function but with checkpoiting"""
    assert lambda_ >= mu, "lambda must be greater or equal to mu."

    checkpointFile = Path(settings['resultsDirMisc'], settings['checkpointFile'])

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
        start_gen = 0    

        logbook = tools.Logbook()
        gradCheck = settings['gradCheck']


        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, map(list, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        avg, bestMin, bestProd = util.averageFitness(population)
        print('avg', avg, 'bestMin', bestMin, 'bestProd', bestProd)

        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        cp = dict(population=population, generation=start_gen, halloffame=halloffame,
            logbook=logbook, rndstate=random.getstate(), gradCheck=gradCheck)

        print('hof', halloffame)

        with checkpointFile.open('wb')as cp_file:
            pickle.dump(cp, cp_file)

    # Begin the generational process
    for gen in range(start_gen, ngen+1):
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, map(list, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("About to start gradient search")
        gradCheck, newChildren = gradFD.search(gradCheck, offspring, cache)
        print("Finished gradient search with new children", len(newChildren))
        offspring.extend(newChildren)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(offspring + population, mu)

        avg, bestMin, bestProd = util.averageFitness(population)
        print('avg', avg, 'bestMin', bestMin, 'bestProd', bestProd)
        
        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        if verbose:
            print(logbook.stream)

        cp = dict(population=population, generation=gen, halloffame=halloffame,
            logbook=logbook, rndstate=random.getstate(), gradCheck=gradCheck)

        print('hof', halloffame)

        hof = Path(settings['resultsDirMisc'], 'hof')
        with hof.open('wb') as data:
            numpy.savetxt(data, numpy.array(halloffame))
        with checkpointFile.open('wb') as cp_file:
            pickle.dump(cp, cp_file)

        if avg > settings['stopAverage'] or bestMin > settings['stopBest']:
            return halloffame
    return halloffame