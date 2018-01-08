import util
import gradFD
import random

name = "Multistart"

def run(cache, tools, creator):
    "run the parameter estimation"
    random.seed()

    parameters = len(cache.MIN_VALUE)

    LAMBDA = parameters * cache.settings['population']

    pop = cache.toolbox.population(n=LAMBDA)

    if "seeds" in cache.settings:
        seed_pop = [cache.toolbox.individual_guess([f(v) for f, v in zip(cache.settings['transform'], sublist)]) for sublist in cache.settings['seeds']]
        pop.extend(seed_pop)

    hof = tools.ParetoFront(similar=util.similar)

    gradCheck = cache.badScore

    gradCheck, newChildren = gradFD.search(gradCheck, pop, cache, check_all=True)

    hof.update(newChildren)

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
