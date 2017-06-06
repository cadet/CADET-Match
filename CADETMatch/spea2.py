import random
import math
import pickle
import util
import numpy
import array
from pathlib import Path
import grad

from deap import algorithms

def run(settings, toolbox, tools, creator):
    "run the parameter estimation"
    random.seed()

    parameters = len(settings['parameters'])

    LAMBDA=parameters * settings['population']
    MU = int(math.ceil(settings['keep']*LAMBDA))
    if MU < 2:
        MU = 2

    pop = toolbox.population(n=LAMBDA)

    totalGenerations = parameters * settings['generations']

    hof = tools.HallOfFame(1)

    eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
        cxpb=settings['crossoverRate'], mutpb=settings['mutationRate'], ngen=totalGenerations, settings=settings, halloffame=hof, tools=tools)

def eaMuCommaLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, settings,
                    stats=None, halloffame=None, verbose=__debug__, tools=None):
    """This is the :math:`(\mu~,~\lambda)` evoslutionary algorithm.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param mu: The number of individuals to select for the next generation.
    :param lambda\_: The number of children to produce at each generation.
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                    inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                        contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
                evolution

    The algorithm takes in a population and evolves it in place using the
    :func:`varOr` function. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statsistics of the evolution. The
    logbook will contain the generation number, the number of evalutions for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varOr` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
            evaluate(offspring)
            population = select(offspring, mu)

    First, the individuals having an invalid fitness are evaluated. Second,
    the evolutionary loop begins by producing *lambda_* offspring from the
    population, the offspring are generated by the :func:`varOr` function. The
    offspring are then evaluated and the next generation population is
    selected from both the offspring **and** the population. Finally, when
    *ngen* generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    .. note::
s
        Care must be taken when the lambda:mu ratio is 1 to 1 as a non-stochastic
        selection will result in no selection at all as
        the operator selects *lambda* individuals from a pool of *mu*.


    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox. This algorithm uses the :func:`varOr`
    variation.
    """
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
        gradCheck = 0.7


        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        cp = dict(population=population, generation=start_gen, halloffame=halloffame,
            logbook=logbook, rndstate=random.getstate(), gradCheck=gradCheck)

        with checkpointFile.open('wb')as cp_file:
            pickle.dump(cp, cp_file)

    # Begin the generational process
    for gen in range(start_gen, ngen+1):
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        gradCheck, newChildren = grad.search(gradCheck, offspring, toolbox)
        offspring.extend(newChildren)

        avg, bestMin = util.averageFitness(offspring)
        print('avg', avg, 'best', bestMin)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        if verbose:
            print(logbook.stream)

        cp = dict(population=population, generation=gen, halloffame=halloffame,
            logbook=logbook, rndstate=random.getstate(), gradCheck=gradCheck)

        hof = Path(settings['resultsDirMisc'], 'hof')
        with hof.open('wb') as data:
            numpy.savetxt(data, numpy.array(halloffame))
        with checkpointFile.open('wb') as cp_file:
            pickle.dump(cp, cp_file)

        if avg > settings['stopAverage'] or bestMin > settings['stopBest']:
            return

def setupDEAP(numGoals, settings, target, MIN_VALUE, MAX_VALUE, fitness, map_function, creator, toolbox, base, tools):
    "setup the DEAP variables"

    creator.create("FitnessMax", base.Fitness, weights=[1.0] * numGoals)
    creator.create("Individual", list, typecode="d", fitness=creator.FitnessMax, strategy=None)

    toolbox.register("individual", util.generateIndividual, creator.Individual,
        len(MIN_VALUE), MIN_VALUE, MAX_VALUE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("individual_guess", util.initIndividual, creator.Individual)

    toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20.0, low=MIN_VALUE, up=MAX_VALUE)
    toolbox.register("mutate", tools.mutPolynomialBounded, eta=20.0, low=MIN_VALUE, up=MAX_VALUE, indpb=1.0/len(MIN_VALUE))

    toolbox.register("select", tools.selSPEA2)
    toolbox.register("evaluate", fitness)

    toolbox.register('map', map_function)
    return toolbox