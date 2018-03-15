import random
import pickle
import util
import numpy
from pathlib import Path
#import grad
import time
import csv

from deap import algorithms
import pareto

name = "NSGA2"

def run(cache, tools, creator):
    "run the parameter estimation"
    random.seed()

    parameters = len(cache.MIN_VALUE)

    populationSize = parameters * cache.settings['population']

    #populationSize has to be a multiple of 4 so increase to the next multiple of 4
    populationSize += (-populationSize % 4)

    CXPB = cache.settings['crossoverRate']

    totalGenerations = parameters * cache.settings['generations']

    checkpointFile = Path(cache.settings['resultsDirMisc'], cache.settings['checkpointFile'])

    path = Path(cache.settings['resultsDirBase'], cache.settings['CSV'])
    with path.open('a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)

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

            population = cache.toolbox.population(n=populationSize)

            if "seeds" in cache.settings:
                seed_pop = [cache.toolbox.individual_guess([f(v) for f, v in zip(cache.settings['transform'], sublist)]) for sublist in cache.settings['seeds']]
                population.extend(seed_pop)

            start_gen = 0    

            halloffame = pareto.ParetoFront(similar=util.similar)
            meta_hof = pareto.ParetoFront(similar=util.similar)
            logbook = tools.Logbook()
            gradCheck = cache.settings['gradCheck']

            logbook.header = ['gen', 'nevals']

        sim_start = generation_start = time.time()
   
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        stalled = util.eval_population(cache.toolbox, cache, invalid_ind, writer, csvfile, halloffame, meta_hof, -1)

        avg, bestMin, bestProd = util.averageFitness(population, cache)
        util.writeProgress(cache, -1, population, halloffame, meta_hof, avg, bestMin, bestProd, sim_start, generation_start)
        util.graph_process(cache)
        
        #if halloffame is not None:
        #    util.updateParetoFront(halloffame, population)

        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        population = cache.toolbox.select(population, len(population)) 
    
        logbook.record(gen=start_gen, evals=len(invalid_ind))

        cp = dict(population=population, generation=start_gen, halloffame=halloffame,
            logbook=logbook, rndstate=random.getstate(), gradCheck=gradCheck, meta_halloffame=meta_hof)
        #cp = dict(population=population, generation=start_gen, halloffame=halloffame,
        #    logbook=logbook, rndstate=random.getstate())

        with checkpointFile.open('wb')as cp_file:
            pickle.dump(cp, cp_file)

        # Begin the generational process
        for gen in range(start_gen, totalGenerations+1):
            generation_start = time.time()
            # Vary the population
            offspring = tools.selTournamentDCD(population, len(population))
            offspring = [cache.toolbox.clone(ind) for ind in offspring]
                
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= CXPB:
                    cache.toolbox.mate(ind1, ind2)
            
                cache.toolbox.mutate(ind1)
                cache.toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values

            offspring = util.RoundOffspring(cache, offspring, halloffame)
        
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            stalled = util.eval_population(cache.toolbox, cache, invalid_ind, writer, csvfile, halloffame, meta_hof, gen)

            gradCheck, newChildren = cache.toolbox.grad_search(gradCheck, offspring, cache, writer, csvfile)
            offspring.extend(newChildren)

            avg, bestMin, bestProd = util.averageFitness(offspring, cache)

            util.writeProgress(cache, gen, offspring, halloffame, meta_hof, avg, bestMin, bestProd, sim_start, generation_start)
            util.graph_process(cache)

            # Select the next generation population
            population = cache.toolbox.select(population + offspring, populationSize)
            logbook.record(gen=gen, evals=len(invalid_ind))

            cp = dict(population=population, generation=start_gen, halloffame=halloffame,
                logbook=logbook, rndstate=random.getstate(), gradCheck=gradCheck, meta_halloffame=meta_hof)

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

def setupDEAP(cache, fitness, grad_fitness, grad_search, map_function, creator, base, tools):
    "setup the DEAP variables"
    creator.create("FitnessMax", base.Fitness, weights=[1.0] * cache.numGoals)
    creator.create("Individual", list, typecode="d", fitness=creator.FitnessMax, strategy=None)

    cache.toolbox.register("individual", util.generateIndividual, creator.Individual,
        len(cache.MIN_VALUE), cache.MIN_VALUE, cache.MAX_VALUE, cache)
    cache.toolbox.register("population", tools.initRepeat, list, cache.toolbox.individual)

    cache.toolbox.register("individual_guess", util.initIndividual, creator.Individual, cache)

    cache.toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20.0, low=cache.MIN_VALUE, up=cache.MAX_VALUE)

    if cache.adaptive:
        cache.toolbox.register("mutate", util.mutationBoundedAdaptive, low=cache.MIN_VALUE, up=cache.MAX_VALUE, indpb=1.0/len(cache.MIN_VALUE))
        cache.toolbox.register("force_mutate", util.mutationBoundedAdaptive, low=cache.MIN_VALUE, up=cache.MAX_VALUE, indpb=1.0)
    else:
        cache.toolbox.register("mutate", tools.mutPolynomialBounded, eta=20.0, low=cache.MIN_VALUE, up=cache.MAX_VALUE, indpb=1.0/len(cache.MIN_VALUE))
        cache.toolbox.register("force_mutate", tools.mutPolynomialBounded, eta=20.0, low=cache.MIN_VALUE, up=cache.MAX_VALUE, indpb=1.0)

    cache.toolbox.register("select", tools.selNSGA2)
    cache.toolbox.register("evaluate", fitness, json_path=cache.json_path)
    cache.toolbox.register("evaluate_grad", grad_fitness, json_path=cache.json_path)
    cache.toolbox.register('grad_search', grad_search)

    cache.toolbox.register('map', map_function)
