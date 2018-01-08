import util
import random

name = 'ScoreTest'

def run(cache, tools, creator):
    "run the parameter estimation"
    random.seed()

    pop = cache.toolbox.population(n=0)

    if "seeds" in cache.settings:
        print(cache.settings['seeds'])
        seed_pop = [cache.toolbox.individual_guess([f(v) for f, v in zip(cache.settings['transform'], sublist)]) for sublist in cache.settings['seeds']]
        pop.extend(seed_pop)
        print(pop)

    hof = tools.ParetoFront(similar=util.similar)

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = cache.toolbox.map(cache.toolbox.evaluate, map(list, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    hof.update(pop)

    return hof

def setupDEAP(cache, fitness, map_function, creator, base, tools):
    "setup the DEAP variables"
    creator.create("FitnessMax", base.Fitness, weights=[1.0] * cache.numGoals)
    creator.create("Individual", list, typecode="d", fitness=creator.FitnessMax, strategy=None)

    cache.toolbox.register("individual", util.generateIndividual, creator.Individual,
        len(cache.MIN_VALUE), cache.MIN_VALUE, cache.MAX_VALUE)
    cache.toolbox.register("population", tools.initRepeat, list, cache.toolbox.individual)

    cache.toolbox.register("individual_guess", util.initIndividual, creator.Individual)

    cache.toolbox.register("evaluate", fitness, json_path=cache.json_path)

    cache.toolbox.register('map', map_function)
