from pathlib import Path
import pickle
import random
import numpy
import util
from deap import algorithms
import time
import csv
import pareto
import scoop

stallRate = 1.25
progressRate = 0.75

def eaMuCommaLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, settings,
                    stats=None, verbose=__debug__, tools=None, cache=None):
    """from DEAP function but with checkpoiting"""

    assert lambda_ >= mu, "lambda must be greater or equal to mu."

    checkpointFile = Path(settings['resultsDirMisc'], settings['checkpointFile'])

    sim_start = generation_start = time.time()

    path = Path(cache.settings['resultsDirBase'], cache.settings['CSV'])
    result_data = {'input':[], 'output':[], 'output_meta':[], 'results':{}, 'times':{}, 'input_transform':[], 'input_transform_extended':[], 'strategy':[], 
                   'mean':[], 'confidence':[]}
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
                stalled, stallWarn, progressWarn = util.eval_population(toolbox, cache, invalid_ind, writer, csvfile, halloffame, meta_hof, -1, result_data)

                avg, bestMin, bestProd = util.averageFitness(population, cache)
                util.writeProgress(cache, -1, population, halloffame, meta_hof, grad_hof, avg, bestMin, bestProd, sim_start, generation_start, result_data)
                util.graph_process(cache, "First")

            cp = dict(population=population, generation=start_gen, halloffame=halloffame,
                rndstate=random.getstate(), gradCheck=gradCheck, meta_halloffame=meta_hof, grad_halloffame=grad_hof)

            with checkpointFile.open('wb')as cp_file:
                pickle.dump(cp, cp_file)

        # Begin the generational process
        for gen in range(start_gen, ngen+1):
            generation_start = time.time()
            # Vary the population
            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb, cache)
            #offspring = util.RoundOffspring(cache, offspring, halloffame)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            stalled, stallWarn, progressWarn = util.eval_population(toolbox, cache, invalid_ind, writer, csvfile, halloffame, meta_hof, gen, result_data)

            gradCheck, newChildren = cache.toolbox.grad_search(gradCheck, offspring, cache, writer, csvfile, grad_hof, meta_hof, gen)
            offspring.extend(newChildren)

            # Select the next generation population
            population[:] = toolbox.select(offspring, mu)

            avg, bestMin, bestProd = util.averageFitness(offspring, cache)
            util.writeProgress(cache, gen, offspring, halloffame, meta_hof, grad_hof, avg, bestMin, bestProd, sim_start, generation_start, result_data)
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
                rndstate=random.getstate(), gradCheck=gradCheck, meta_halloffame=meta_hof, grad_halloffame=grad_hof)

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
    result_data = {'input':[], 'output':[], 'output_meta':[], 'results':{}, 'times':{}, 'input_transform':[], 'input_transform_extended':[], 'strategy':[], 
                   'mean':[], 'confidence':[]}
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
                stalled, stallWarn, progressWarn = util.eval_population(toolbox, cache, invalid_ind, writer, csvfile, halloffame, meta_hof, -1, result_data)

                avg, bestMin, bestProd = util.averageFitness(population, cache)
                util.writeProgress(cache, -1, population, halloffame, meta_hof, grad_hof, avg, bestMin, bestProd, sim_start, generation_start, result_data)
                util.graph_process(cache, "First")

            cp = dict(population=population, generation=start_gen, halloffame=halloffame,
                rndstate=random.getstate(), gradCheck=gradCheck, meta_halloffame=meta_hof, grad_halloffame=grad_hof)

            with checkpointFile.open('wb')as cp_file:
                pickle.dump(cp, cp_file)

        # Begin the generational process
        for gen in range(start_gen, ngen+1):
            generation_start = time.time()
            # Vary the population
            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb, cache)
            #offspring = util.RoundOffspring(cache, offspring, halloffame)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            stalled, stallWarn, progressWarn = util.eval_population(toolbox, cache, invalid_ind, writer, csvfile, halloffame, meta_hof, gen, result_data)

            # Combination of varOr and RoundOffSpring invalidates some members of the population, not sure why yet
            #invalid_ind = [ind for ind in population if not ind.fitness.valid]
            #scoop.logger.info("we have invalid population %s", len(invalid_ind))
            #stalled, stallWarn, progressWarn = util.eval_population(toolbox, cache, invalid_ind, writer, csvfile, halloffame, meta_hof, gen, result_data)

            gradCheck, newChildren = cache.toolbox.grad_search(gradCheck, offspring, cache, writer, csvfile, grad_hof, meta_hof, gen)
            offspring.extend(newChildren)

            avg, bestMin, bestProd = util.averageFitness(offspring, cache)
            
            # Select the next generation population
            population[:] = toolbox.select(offspring + population, mu)

            util.writeProgress(cache, gen, offspring, halloffame, meta_hof, grad_hof, avg, bestMin, bestProd, sim_start, generation_start, result_data)
            util.graph_process(cache, gen)

            cp = dict(population=population, generation=gen, halloffame=halloffame,
                rndstate=random.getstate(), gradCheck=gradCheck, meta_halloffame=meta_hof, grad_halloffame=grad_hof)

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

def nsga2(populationSize, ngen, cache, tools):
    """NSGA2 with checkpointing"""
    #fix max population to be a multiple of 4
    cache.settings['maxPopulation'] = cache.settings['maxPopulation'] + (-cache.settings['maxPopulation'] % 4)

    cxpb = cache.settings['crossoverRate']
    checkpointFile = Path(cache.settings['resultsDirMisc'], cache.settings['checkpointFile'])

    path = Path(cache.settings['resultsDirBase'], cache.settings['CSV'])
    result_data = {'input':[], 'output':[], 'output_meta':[], 'results':{}, 'times':{}, 'input_transform':[], 'input_transform_extended':[], 'strategy':[], 
                   'mean':[], 'confidence':[]}
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
            stalled, stallWarn, progressWarn = util.eval_population(cache.toolbox, cache, invalid_ind, writer, csvfile, halloffame, meta_hof, -1, result_data)
            stalled, stallWarn, progressWarn = util.eval_population(cache.toolbox, cache, invalid_ind, writer, csvfile, halloffame, meta_hof, -1, result_data)

            avg, bestMin, bestProd = util.averageFitness(population, cache)
            util.writeProgress(cache, -1, population, halloffame, meta_hof, grad_hof, avg, bestMin, bestProd, sim_start, generation_start, result_data)
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
            stalled, stallWarn, progressWarn = util.eval_population(cache.toolbox, cache, invalid_ind, writer, csvfile, halloffame, meta_hof, gen, result_data)

            gradCheck, newChildren = cache.toolbox.grad_search(gradCheck, offspring, cache, writer, csvfile, grad_hof, meta_hof, gen)
            offspring.extend(newChildren)

            avg, bestMin, bestProd = util.averageFitness(offspring, cache)

            util.writeProgress(cache, gen, offspring, halloffame, meta_hof, grad_hof, avg, bestMin, bestProd, sim_start, generation_start, result_data)
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
                util.eval_population(cache.toolbox, cache, invalid_ind, writer, csvfile, halloffame, meta_hof, gen, result_data)

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

def varOr(population, toolbox, lambda_, cxpb, mutpb, cache):
    """This version of varOr has been taken from DEAP and modified to create
    lambda_ unique individuals based on roundParameters. This should improve population
    diversity while also eliminating near duplicate individuals.
    """
    """Part of an evolutionary algorithm applying only the variation part
    (crossover, mutation **or** reproduction). The modified individuals have
    their fitness invalidated. The individuals are cloned so returned
    population is independent of the input population.
    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param lambda\_: The number of children to produce
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: The final population
    :returns: A :class:`~deap.tools.Logbook` with the statistics of the
              evolution
    The variation goes as follow. On each of the *lambda_* iteration, it
    selects one of the three operations; crossover, mutation or reproduction.
    In the case of a crossover, two individuals are selected at random from
    the parental population :math:`P_\mathrm{p}`, those individuals are cloned
    using the :meth:`toolbox.clone` method and then mated using the
    :meth:`toolbox.mate` method. Only the first child is appended to the
    offspring population :math:`P_\mathrm{o}`, the second child is discarded.
    In the case of a mutation, one individual is selected at random from
    :math:`P_\mathrm{p}`, it is cloned and then mutated using using the
    :meth:`toolbox.mutate` method. The resulting mutant is appended to
    :math:`P_\mathrm{o}`. In the case of a reproduction, one individual is
    selected at random from :math:`P_\mathrm{p}`, cloned and appended to
    :math:`P_\mathrm{o}`.
    This variation is named *Or* beceause an offspring will never result from
    both operations crossover and mutation. The sum of both probabilities
    shall be in :math:`[0, 1]`, the reproduction probability is
    1 - *cxpb* - *mutpb*.
    """

    #If we are operating with unlimited precision there is no need to trim
    if cache.roundParameters is None:
        return algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")

    RoundChild = util.RoundChild

    offspring = []
    unique = set()
    while len(offspring) < lambda_:
        op_choice = random.random()
        if op_choice < cxpb:            # Apply crossover
            ind1, ind2 = map(toolbox.clone, random.sample(population, 2))
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values

            RoundChild(cache, ind1)
            key = tuple(ind1)
            if key not in unique:
                offspring.append(ind1)
                unique.add(key)

        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)
            del ind.fitness.values

            RoundChild(cache, ind)
            key = tuple(ind)
            if key not in unique:
                offspring.append(ind)
                unique.add(key)
        else:                           # Apply reproduction
            ind = toolbox.clone(random.choice(population))
            util.RoundChild(cache, ind)
            key = tuple(ind)
            if key not in unique:
                offspring.append(ind)
                unique.add(key)

            #offspring.append(random.choice(population))

    return offspring
