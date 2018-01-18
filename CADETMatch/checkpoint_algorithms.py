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
                    stats=None, halloffame=None, verbose=__debug__, tools=None, cache=None):
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
            for ind, result in zip(invalid_ind, fitnesses):
                fit, csv_line = result
                ind.fitness.values = fit
                writer.writerow(csv_line)
                csvfile.flush()

            if halloffame is not None:
                halloffame.update(population)

            avg, bestMin, bestProd = util.averageFitness(population)
            util.writeProgress(cache, -1, population, halloffame, avg, bestMin, bestProd, sim_start, generation_start)

            logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=0, nevals=len(invalid_ind), **record)

            cp = dict(population=population, generation=start_gen, halloffame=halloffame,
                logbook=logbook, rndstate=random.getstate(), gradCheck=gradCheck)

            with checkpointFile.open('wb')as cp_file:
                pickle.dump(cp, cp_file)
            util.space_plots(cache)

        # Begin the generational process
        for gen in range(start_gen, ngen+1):
            generation_start = time.time()
            # Vary the population
            offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)
            offspring = util.RoundOffspring(cache, offspring)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, map(list, invalid_ind))
            for ind, result in zip(invalid_ind, fitnesses):
                fit, csv_line = result
                ind.fitness.values = fit
                writer.writerow(csv_line)
                csvfile.flush()

            gradCheck, newChildren = gradFD.search(gradCheck, offspring, cache)
            offspring.extend(newChildren)

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Select the next generation population
            population[:] = toolbox.select(offspring, mu)

            avg, bestMin, bestProd = util.averageFitness(offspring)
            util.writeProgress(cache, gen, offspring, halloffame, avg, bestMin, bestProd, sim_start, generation_start)
        
            # Update the statistics with the new population
            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)

            cp = dict(population=population, generation=gen, halloffame=halloffame,
                logbook=logbook, rndstate=random.getstate(), gradCheck=gradCheck)

            hof = Path(settings['resultsDirMisc'], 'hof')
            with hof.open('wb') as data:
                numpy.savetxt(data, numpy.array(halloffame))
            with checkpointFile.open('wb') as cp_file:
                pickle.dump(cp, cp_file)

            util.space_plots(cache)

            if avg > settings['stopAverage'] or bestMin > settings['stopBest']:
                return halloffame
        return halloffame


def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, settings,
                   stats=None, halloffame=None, verbose=__debug__, tools=None, cache=None):
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
            for ind, result in zip(invalid_ind, fitnesses):
                fit, csv_line = result
                ind.fitness.values = fit
                writer.writerow(csv_line)
                csvfile.flush()

            if halloffame is not None:
                halloffame.update(population)

            avg, bestMin, bestProd = util.averageFitness(population)
            util.writeProgress(cache, -1, population, halloffame, avg, bestMin, bestProd, sim_start, generation_start)

            logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=0, nevals=len(invalid_ind), **record)

            cp = dict(population=population, generation=start_gen, halloffame=halloffame,
                logbook=logbook, rndstate=random.getstate(), gradCheck=gradCheck)

            with checkpointFile.open('wb')as cp_file:
                pickle.dump(cp, cp_file)

            util.space_plots(cache)

        # Begin the generational process
        for gen in range(start_gen, ngen+1):

            generation_start = time.time()
            # Vary the population
            offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)
            offspring = util.RoundOffspring(cache, offspring)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, map(list, invalid_ind))
            for ind, result in zip(invalid_ind, fitnesses):
                fit, csv_line = result
                ind.fitness.values = fit
                writer.writerow(csv_line)
                csvfile.flush()

            # Combination of varOr and RoundOffSpring invalidates some members of the population, not sure why yet
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, map(list, invalid_ind))
            for ind, result in zip(invalid_ind, fitnesses):
                fit, csv_line = result
                ind.fitness.values = fit
                writer.writerow(csv_line)
                csvfile.flush()

            gradCheck, newChildren = gradFD.search(gradCheck, offspring, cache)
            offspring.extend(newChildren)

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Select the next generation population
            population[:] = toolbox.select(offspring + population, mu)

            avg, bestMin, bestProd = util.averageFitness(offspring)
            util.writeProgress(cache, gen, offspring, halloffame, avg, bestMin, bestProd, sim_start, generation_start)
        
            # Update the statistics with the new population
            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)

            cp = dict(population=population, generation=gen, halloffame=halloffame,
                logbook=logbook, rndstate=random.getstate(), gradCheck=gradCheck)

            hof = Path(settings['resultsDirMisc'], 'hof')
            with hof.open('wb') as data:
                numpy.savetxt(data, numpy.array(halloffame))
            with checkpointFile.open('wb') as cp_file:
                pickle.dump(cp, cp_file)

            util.space_plots(cache)

            if avg > settings['stopAverage'] or bestMin > settings['stopBest']:
                return halloffame
        return halloffame
