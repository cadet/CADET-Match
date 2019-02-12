#This is a simple implemtnation of NSGA2 that runs on a single thread and has the same kind of interface as the scipy optimize tools
#The reason for this is that finding the maximum likelihood and bandwidth works better with NSGA2 and the optimizers built into scipy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import SALib.sample.sobol_sequence
import numpy
import random
import array

def sobolGenerator(icls, dimension, lb, ub, n):
    ub = numpy.array(ub)
    lb = numpy.array(lb)
    if n > 0:
        populationDimension = dimension
        populationSize = n
        sobol = SALib.sample.sobol_sequence.sample(populationSize, populationDimension)
        data = numpy.apply_along_axis(list, 1, sobol) * (ub-lb) + lb
        data = list(map(icls, data))
        return data
    else:
        return []

def nsga2_min(func, lb, ub, ngen=8000, mu=200, cxpb=0.8, args=None, stop=40):
    FitnessMin = create("FitnessMin", base.Fitness, weights=(-1.0,))
    Individual = create("Individual", list, fitness=FitnessMin)

    toolbox = base.Toolbox()

    toolbox.register("evaluate", func, *args)
    toolbox.register("population", sobolGenerator, Individual, len(lb), lb, ub)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=lb, up=ub, eta=2.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=lb, up=ub, eta=1.0, indpb=1.0/len(lb))
    toolbox.register("select", tools.selNSGA2)

    toolbox.register('map', map)

    pop = toolbox.population(n=mu)

    best = []
    best_score = 1e308
    last_progress = 0

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
        if fit[0] < best_score:
            best_score = fit[0]
            best = ind

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    # Begin the generational process
    for gen in range(1, ngen):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= cxpb:
                toolbox.mate(ind1, ind2)
            
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

            if fit[0] < best_score:
                best_score = fit[0]
                best = ind
                last_progress = gen


        if gen - last_progress > stop:
            print("stopped at gen %s" % gen)
            break

        # Select the next generation population
        pop = toolbox.select(pop + offspring, mu)

        print('best_score\t', best_score, "\tbest\t", best)
        
    result = {}
    result['x'] = best
    result['fun'] = best_score
    result['pop'] = pop

    return result


#From DEAP and modified, need to get rid of this entirely and figure out how to just build the class
#I modified this version so it does not create a global class
def create(name, base, **kargs):
    dict_inst = {}
    dict_cls = {}
    for obj_name, obj in kargs.items():
        if isinstance(obj, type):
            dict_inst[obj_name] = obj
        else:
            dict_cls[obj_name] = obj

    def initType(self, *args, **kargs):
        """Replace the __init__ function of the new type, in order to
        add attributes that were defined with **kargs to the instance.
        """
        for obj_name, obj in dict_inst.items():
            setattr(self, obj_name, obj())
        if base.__init__ is not object.__init__:
            base.__init__(self, *args, **kargs)

    objtype = type(str(name), (base,), dict_cls)
    objtype.__init__ = initType
    return objtype