import random
import math
import pickle
import util
import numpy
import array
from pathlib import Path
#import grad
import evo
import gradFD

from deap import algorithms

def run(settings, toolbox, tools, creator):
    "run the parameter estimation"
    random.seed()

    parameters = len(evo.MIN_VALUE)

    populationSize=parameters * settings['population']
    CXPB=settings['crossoverRate']

    totalGenerations = parameters * settings['generations']

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

        population = toolbox.population(n=populationSize)

        if "seeds" in settings:
            seed_pop = [toolbox.individual_guess([f(v) for f, v in zip(settings['transform'], sublist)]) for sublist in settings['seeds']]
            population.extend(seed_pop)

        start_gen = 0    

        halloffame = tools.ParetoFront()
        logbook = tools.Logbook()
        gradCheck = settings['gradCheck']

        logbook.header = ['gen', 'nevals']
   
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit    

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    population = toolbox.select(population, len(population)) 
    
    avg, bestMin = util.averageFitness(population)
    print('avg', avg, 'best', bestMin)

    logbook.record(gen=start_gen, evals=len(invalid_ind))

    cp = dict(population=population, generation=start_gen, halloffame=halloffame,
        logbook=logbook, rndstate=random.getstate(), gradCheck=gradCheck)
    #cp = dict(population=population, generation=start_gen, halloffame=halloffame,
    #    logbook=logbook, rndstate=random.getstate())

    with checkpointFile.open('wb')as cp_file:
        pickle.dump(cp, cp_file)

    # Begin the generational process
    for gen in range(start_gen, totalGenerations+1):
        # Vary the population
        offspring = tools.selTournamentDCD(population, len(population))
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)
            
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("About to start gradient search")
        gradCheck, newChildren = gradFD.search(gradCheck, offspring, toolbox)
        print("Finished gradient search with new children", len(newChildren))
        offspring.extend(newChildren)

        avg, bestMin = util.averageFitness(offspring)
        print('avg', avg, 'best', bestMin)

        # Select the next generation population
        population = toolbox.select(population + offspring, populationSize)
        logbook.record(gen=gen, evals=len(invalid_ind))

        #cp = dict(population=population, generation=gen, halloffame=halloffame,
        #    logbook=logbook, rndstate=random.getstate())

        cp = dict(population=population, generation=start_gen, halloffame=halloffame,
            logbook=logbook, rndstate=random.getstate(), gradCheck=gradCheck)

        hof = Path(settings['resultsDirMisc'], 'hof')
        with hof.open('wb') as data:
            numpy.savetxt(data, numpy.array(halloffame))
        with checkpointFile.open('wb') as cp_file:
            pickle.dump(cp, cp_file)

        if avg > settings['stopAverage'] or bestMin > settings['stopBest']:
            return halloffame
    return halloffame

def setupDEAP(numGoals, settings, target, MIN_VALUE, MAX_VALUE, fitness, map_function, creator, toolbox, base, tools):
    "setup the DEAP variables"

    creator.create("FitnessMax", base.Fitness, weights=[1.0] * numGoals)
    creator.create("Individual", list, typecode="d", fitness=creator.FitnessMax, strategy=None)

    toolbox.register("individual", util.generateIndividual, creator.Individual,
        len(MIN_VALUE), MIN_VALUE, MAX_VALUE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("individual_guess", util.initIndividual, creator.Individual)

    toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20.0, low=MIN_VALUE, up=MAX_VALUE)
    toolbox.register("mutate", util.mutPolynomialBoundedAdaptive, eta=10.0, low=MIN_VALUE, up=MAX_VALUE, indpb=1.0/len(MIN_VALUE))

    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", fitness)

    toolbox.register('map', map_function)
    return toolbox