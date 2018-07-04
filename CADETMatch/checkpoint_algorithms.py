from pathlib import Path
import pickle
import random
import numpy
import util
from deap import algorithms
import time
import csv
import pareto

stallRate = 1.25
progressRate = 0.75

def eaMuCommaLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, settings,
                    stats=None, verbose=__debug__, tools=None, cache=None):
    """from DEAP function but with checkpoiting"""

    assert lambda_ >= mu, "lambda must be greater or equal to mu."

    checkpointFile = Path(settings['resultsDirMisc'], settings['checkpointFile'])

    sim_start = generation_start = time.time()

    path = Path(cache.settings['resultsDirBase'], cache.settings['CSV'])
    training = {'input':[], 'output':[], 'output_meta':[], 'results':{}, 'times':{}}
    with path.open('a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)

        if checkpointFile.exists():
            with checkpointFile.open('rb') as cp_file:
                cp = pickle.load(cp_file)
            population = cp["population"]
            start_gen = cp["generation"]    
    
            halloffame = cp["halloffame"]
            meta_hof = cp['meta_halloffame']
            grad_hof = cp['grad_halloffame']
            random.setstate(cp["rndstate"])

            if cp['gradCheck'] > cache.settings['gradCheck']:
                gradCheck = cp['gradCheck']
            else:
                gradCheck = cache.settings['gradCheck']

        else:
            # Start a new evolution
            start_gen = 0    

            gradCheck = settings['gradCheck']

            halloffame = pareto.ParetoFront(similar=util.similar)
            meta_hof = pareto.ParetoFront(similar=util.similar)
            grad_hof = pareto.ParetoFront(similar=util.similar)


            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            if invalid_ind:
                stalled, stallWarn, progressWarn = util.eval_population(toolbox, cache, invalid_ind, writer, csvfile, halloffame, meta_hof, -1, training)

                avg, bestMin, bestProd = util.averageFitness(population, cache)
                util.writeProgress(cache, -1, population, halloffame, meta_hof, grad_hof, avg, bestMin, bestProd, sim_start, generation_start, training)
                util.graph_process(cache, "First")

            cp = dict(population=population, generation=start_gen, halloffame=halloffame,
                rndstate=random.getstate(), gradCheck=gradCheck, meta_halloffame=meta_hof)

            with checkpointFile.open('wb')as cp_file:
                pickle.dump(cp, cp_file)

        # Begin the generational process
        for gen in range(start_gen, ngen+1):
            generation_start = time.time()
            # Vary the population
            offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)
            offspring = util.RoundOffspring(cache, offspring, halloffame)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            stalled, stallWarn, progressWarn = util.eval_population(toolbox, cache, invalid_ind, writer, csvfile, halloffame, meta_hof, gen, training)

            gradCheck, newChildren = cache.toolbox.grad_search(gradCheck, offspring, cache, writer, csvfile, grad_hof, meta_hof, gen)
            offspring.extend(newChildren)

            # Select the next generation population
            population[:] = toolbox.select(offspring, mu)

            avg, bestMin, bestProd = util.averageFitness(offspring, cache)
            util.writeProgress(cache, gen, offspring, halloffame, meta_hof, grad_hof, avg, bestMin, bestProd, sim_start, generation_start, training)
            util.graph_process(cache, gen)

            if stallWarn:
                maxPopulation = cache.settings['maxPopulation'] * len(cache.MIN_VALUE)
                newLambda_ = int(lambda_ * stallRate)
                lambda_ = min(newLambda_, maxPopulation)

            if progressWarn:
                minPopulation = cache.settings['minPopulation'] * len(cache.MIN_VALUE)
                newLambda_ = int(lambda_ * progressRate)
                lambda_ = max(newLambda_, minPopulation)
                                   
            cp = dict(population=population, generation=gen, halloffame=halloffame,
                rndstate=random.getstate(), gradCheck=gradCheck, meta_halloffame=meta_hof)

            hof = Path(settings['resultsDirMisc'], 'hof')
            with hof.open('wb') as data:
                numpy.savetxt(data, numpy.array(halloffame))
            with checkpointFile.open('wb') as cp_file:
                pickle.dump(cp, cp_file)

            if avg >= settings['stopAverage'] or bestMin >= settings['stopBest'] or stalled:
                util.finish(cache)
                return halloffame
        util.finish(cache)
        return halloffame


def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, settings,
                   stats=None, verbose=__debug__, tools=None, cache=None):
    """from DEAP function but with checkpoiting"""
    assert lambda_ >= mu, "lambda must be greater or equal to mu."

    #import search.spea2
    #from line_profiler import LineProfiler
    #profile = LineProfiler(search.spea2.selSPEA2)

    checkpointFile = Path(settings['resultsDirMisc'], settings['checkpointFile'])

    sim_start = generation_start = time.time()

    path = Path(cache.settings['resultsDirBase'], cache.settings['CSV'])
    training = {'input':[], 'output':[], 'output_meta':[], 'results':{}, 'times':{}}
    with path.open('a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)

        if checkpointFile.exists():
            with checkpointFile.open('rb') as cp_file:
                cp = pickle.load(cp_file)
            population = cp["population"]
            start_gen = cp["generation"]    
    
            halloffame = cp["halloffame"]
            meta_hof = cp['meta_halloffame']
            grad_hof = cp['grad_halloffame']
            random.setstate(cp["rndstate"])
            
            if cp['gradCheck'] > cache.settings['gradCheck']:
                gradCheck = cp['gradCheck']
            else:
                gradCheck = cache.settings['gradCheck']
        else:
            # Start a new evolution
            start_gen = 0    

            gradCheck = settings['gradCheck']

            halloffame = pareto.ParetoFront(similar=util.similar)
            meta_hof = pareto.ParetoFront(similar=util.similar)
            grad_hof = pareto.ParetoFront(similar=util.similar)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            if invalid_ind:
                stalled, stallWarn, progressWarn = util.eval_population(toolbox, cache, invalid_ind, writer, csvfile, halloffame, meta_hof, -1, training)

                avg, bestMin, bestProd = util.averageFitness(population, cache)
                util.writeProgress(cache, -1, population, halloffame, meta_hof, grad_hof, avg, bestMin, bestProd, sim_start, generation_start, training)
                util.graph_process(cache, "First")

            cp = dict(population=population, generation=start_gen, halloffame=halloffame,
                rndstate=random.getstate(), gradCheck=gradCheck, meta_halloffame=meta_hof)

            with checkpointFile.open('wb')as cp_file:
                pickle.dump(cp, cp_file)

        # Begin the generational process
        for gen in range(start_gen, ngen+1):
            generation_start = time.time()
            # Vary the population
            offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)
            offspring = util.RoundOffspring(cache, offspring, halloffame)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            stalled, stallWarn, progressWarn = util.eval_population(toolbox, cache, invalid_ind, writer, csvfile, halloffame, meta_hof, gen, training)

            # Combination of varOr and RoundOffSpring invalidates some members of the population, not sure why yet
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            stalled, stallWarn, progressWarn = util.eval_population(toolbox, cache, invalid_ind, writer, csvfile, halloffame, meta_hof, gen, training)

            gradCheck, newChildren = cache.toolbox.grad_search(gradCheck, offspring, cache, writer, csvfile, grad_hof, meta_hof, gen)
            offspring.extend(newChildren)

            avg, bestMin, bestProd = util.averageFitness(offspring, cache)
            
            # Select the next generation population
            population[:] = toolbox.select(offspring + population, mu)

            util.writeProgress(cache, gen, offspring, halloffame, meta_hof, grad_hof, avg, bestMin, bestProd, sim_start, generation_start, training)
            util.graph_process(cache, gen)

            cp = dict(population=population, generation=gen, halloffame=halloffame,
                rndstate=random.getstate(), gradCheck=gradCheck, meta_halloffame=meta_hof)

            if stallWarn:
                maxPopulation = cache.settings['maxPopulation'] * len(cache.MIN_VALUE)
                newLambda_ = int(lambda_ * stallRate)
                lambda_ = min(newLambda_, maxPopulation)

            if progressWarn:
                minPopulation = cache.settings['minPopulation'] * len(cache.MIN_VALUE)
                newLambda_ = int(lambda_ * progressRate)
                lambda_ = max(newLambda_, minPopulation)


            hof = Path(settings['resultsDirMisc'], 'hof')
            with hof.open('wb') as data:
                numpy.savetxt(data, numpy.array(halloffame))
            with checkpointFile.open('wb') as cp_file:
                pickle.dump(cp, cp_file)

            if avg >= settings['stopAverage'] or bestMin >= settings['stopBest'] or stalled:
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

def nsga2(populationSize, ngen, cache, tools):
    """NSGA2 with checkpointing"""
    #fix max population to be a multiple of 4
    cache.settings['maxPopulation'] = cache.settings['maxPopulation'] + (-cache.settings['maxPopulation'] % 4)

    cxpb = cache.settings['crossoverRate']
    checkpointFile = Path(cache.settings['resultsDirMisc'], cache.settings['checkpointFile'])

    path = Path(cache.settings['resultsDirBase'], cache.settings['CSV'])
    training = {'input':[], 'output':[], 'output_meta':[], 'results':{}, 'times':{}}
    with path.open('a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)

        if checkpointFile.exists():
            with checkpointFile.open('rb') as cp_file:
                cp = pickle.load(cp_file)
            population = cp["population"]
            start_gen = cp["generation"]    
    
            halloffame = cp["halloffame"]
            meta_hof = cp['meta_halloffame']
            grad_hof = cp['grad_halloffame']
            random.setstate(cp["rndstate"])

            if cp['gradCheck'] > cache.settings['gradCheck']:
                gradCheck = cp['gradCheck']
            else:
                gradCheck = cache.settings['gradCheck']

        else:
            # Start a new evolution

            population = cache.toolbox.population(n=populationSize)

            if "seeds" in cache.settings:
                seed_pop = [cache.toolbox.individual_guess([f(v) for f, v in zip(cache.settings['transform'], sublist)]) for sublist in cache.settings['seeds']]
                population.extend(seed_pop)

            start_gen = 0    

            halloffame = pareto.ParetoFront(similar=util.similar)
            meta_hof = pareto.ParetoFront(similar=util.similar)
            grad_hof = pareto.ParetoFront(similar=util.similar)
            gradCheck = cache.settings['gradCheck']


        sim_start = generation_start = time.time()
   
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        if invalid_ind:
            stalled, stallWarn, progressWarn = util.eval_population(cache.toolbox, cache, invalid_ind, writer, csvfile, halloffame, meta_hof, -1, training)
            stalled, stallWarn, progressWarn = util.eval_population(cache.toolbox, cache, invalid_ind, writer, csvfile, halloffame, meta_hof, -1, training)

            avg, bestMin, bestProd = util.averageFitness(population, cache)
            util.writeProgress(cache, -1, population, halloffame, meta_hof, grad_hof, avg, bestMin, bestProd, sim_start, generation_start, training)
            util.graph_process(cache, "First")
        
        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        population = cache.toolbox.select(population, len(population)) 
    

        cp = dict(population=population, generation=start_gen, halloffame=halloffame,
            rndstate=random.getstate(), gradCheck=gradCheck, meta_halloffame=meta_hof, grad_halloffame=grad_hof)

        with checkpointFile.open('wb')as cp_file:
            pickle.dump(cp, cp_file)

        # Begin the generational process
        for gen in range(start_gen, ngen+1):
            generation_start = time.time()
            # Vary the population
            offspring = tools.selTournamentDCD(population, len(population))
            offspring = [cache.toolbox.clone(ind) for ind in offspring]
                
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= cxpb:
                    cache.toolbox.mate(ind1, ind2)
            
                cache.toolbox.mutate(ind1)
                cache.toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values

            offspring = util.RoundOffspring(cache, offspring, halloffame)
        
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            stalled, stallWarn, progressWarn = util.eval_population(cache.toolbox, cache, invalid_ind, writer, csvfile, halloffame, meta_hof, gen, training)

            gradCheck, newChildren = cache.toolbox.grad_search(gradCheck, offspring, cache, writer, csvfile, grad_hof, meta_hof, gen)
            offspring.extend(newChildren)

            avg, bestMin, bestProd = util.averageFitness(offspring, cache)

            util.writeProgress(cache, gen, offspring, halloffame, meta_hof, grad_hof, avg, bestMin, bestProd, sim_start, generation_start, training)
            util.graph_process(cache, gen)

            # Select the next generation population
            population = cache.toolbox.select(population + offspring, populationSize)

            if stallWarn:
                maxPopulation = cache.settings['maxPopulation'] * len(cache.MIN_VALUE)
                newPopulationSize = int(populationSize * stallRate)
                newPopulationSize += (-newPopulationSize % 4)
                newPopulationSize = min(newPopulationSize, maxPopulation)
                newPopulationSize += (-newPopulationSize % 4)

                diffSize = newPopulationSize - populationSize
                newPopulation = cache.toolbox.randomPopulation(n=diffSize)

                invalid_ind = [ind for ind in newPopulation if not ind.fitness.valid]
                util.eval_population(cache.toolbox, cache, invalid_ind, writer, csvfile, halloffame, meta_hof, gen, training)

                # This is just to assign the crowding distance to the individuals
                # no actual selection is done
                newPopulation = cache.toolbox.select(newPopulation, len(newPopulation)) 
                
                population.extend(newPopulation)
                populationSize = newPopulationSize

            if progressWarn:
                minPopulation = cache.settings['minPopulation'] * len(cache.MIN_VALUE)
                newPopulationSize = int(populationSize * progressRate)
                newPopulationSize += (-newPopulationSize % 4)
                newPopulationSize = max(newPopulationSize, minPopulation)
                newPopulationSize += (-newPopulationSize % 4)

                diffSize = populationSize - newPopulationSize
                
                population = cache.toolbox.select(population, newPopulationSize) 
                populationSize = newPopulationSize

            cp = dict(population=population, generation=gen, halloffame=halloffame,
                rndstate=random.getstate(), gradCheck=gradCheck, meta_halloffame=meta_hof, grad_halloffame=grad_hof)

            hof = Path(cache.settings['resultsDirMisc'], 'hof')
            with hof.open('wb') as data:
                numpy.savetxt(data, numpy.array(halloffame))
            with checkpointFile.open('wb') as cp_file:
                pickle.dump(cp, cp_file)

            if avg >= cache.settings['stopAverage'] or bestMin >= cache.settings['stopBest'] or stalled:
                util.finish(cache)
                return halloffame

        util.finish(cache)
        return halloffame