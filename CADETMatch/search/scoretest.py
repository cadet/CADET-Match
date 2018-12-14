import util
import pareto
import csv
from pathlib import Path
import time
import jacobian
import scoop

name = 'ScoreTest'

def run(cache, tools, creator):
    "run the parameter estimation"
    path = Path(cache.settings['resultsDirBase'], cache.settings['CSV'])
    with path.open('a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
        pop = cache.toolbox.population(n=0)
        sim_start = generation_start = time.time()
        result_data = {'input':[], 'output':[], 'output_meta':[], 'results':{}, 'times':{}, 'input_transform':[], 'input_transform_extended':[]}

        if "seeds" in cache.settings:
            seed_pop = [cache.toolbox.individual_guess([f(v) for f, v in zip(cache.settings['transform'], sublist)]) for sublist in cache.settings['seeds']]
            pop.extend(seed_pop)

        hof = pareto.ParetoFront(similar=util.similar)
        meta_hof = pareto.ParetoFront(similar=util.similar)
        grad_hof = pareto.ParetoFront(similar=util.similar)

        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        stalled, stallWarn, progressWarn = util.eval_population(cache.toolbox, cache, invalid_ind, writer, csvfile, hof, meta_hof, -1, result_data)

        if cache.settings.get('condTest' , None):
            for ind in invalid_ind:
                J = jacobian.jac(ind, cache)
                scoop.logger.info('%s %s', ind, J)
        
        avg, bestMin, bestProd = util.averageFitness(pop, cache)
        
        util.writeProgress(cache, -1, pop, hof, meta_hof, grad_hof, avg, bestMin, bestProd, sim_start, generation_start, result_data)
        
        util.finish(cache)
        return hof

def setupDEAP(cache, fitness, grad_fitness, grad_search, map_function, creator, base, tools):
    "setup the DEAP variables"
    creator.create("FitnessMax", base.Fitness, weights=[1.0] * cache.numGoals)
    creator.create("Individual", list, typecode="d", fitness=creator.FitnessMax, strategy=None)

    creator.create("FitnessMaxMeta", base.Fitness, weights=[1.0] * 4)
    creator.create("IndividualMeta", array.array, typecode="d", fitness=creator.FitnessMaxMeta, strategy=None)
    cache.toolbox.register("individualMeta", util.initIndividual, creator.IndividualMeta, cache)

    cache.toolbox.register("individual", util.generateIndividual, creator.Individual,
        len(cache.MIN_VALUE), cache.MIN_VALUE, cache.MAX_VALUE, cache)
    cache.toolbox.register("population", tools.initRepeat, list, cache.toolbox.individual)

    cache.toolbox.register("individual_guess", util.initIndividual, creator.Individual, cache)

    cache.toolbox.register("evaluate", fitness, json_path=cache.json_path)
    cache.toolbox.register("evaluate_grad", grad_fitness, json_path=cache.json_path)
    cache.toolbox.register('grad_search', grad_search)

    cache.toolbox.register('map', map_function)
