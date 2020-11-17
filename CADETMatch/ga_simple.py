# This is a simple implemtnation of NSGA2 that runs on a single thread and has the same kind of interface as the scipy optimize tools
# The reason for this is that finding the maximum likelihood and bandwidth works better with NSGA2 and the optimizers built into scipy

import array
import multiprocessing
import random

import numpy
import SALib.sample.sobol_sequence
from addict import Dict
from deap import algorithms, base, creator, tools


def sobolGenerator(icls, dimension, lb, ub, n):
    ub = numpy.array(ub)
    lb = numpy.array(lb)
    if n > 0:
        populationDimension = dimension
        populationSize = n
        sobol = SALib.sample.sobol_sequence.sample(populationSize, populationDimension)
        data = numpy.apply_along_axis(list, 1, sobol) * (ub - lb) + lb
        data = list(map(icls, data))
        return data
    else:
        return []


def ga_min(func, lb, ub, ngen=500, mu=200, args=None, stop=40):
    FitnessMin = create(
        "FitnessMin",
        base.Fitness,
        weights=[
            -1.0,
        ],
    )
    Individual = create("Individual", list, fitness=FitnessMin)

    toolbox = base.Toolbox()

    if args is not None:
        toolbox.register("evaluate", func, *args)
    else:
        toolbox.register("evaluate", func)
    toolbox.register("population", sobolGenerator, Individual, len(lb), lb, ub)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=lb, up=ub, eta=30.0)
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        low=lb,
        up=ub,
        eta=20.0,
        indpb=1.0 / len(lb),
    )

    ref_points = tools.uniform_reference_points(1, 16)
    toolbox.register("select", tools.selSPEA2)

    toolbox.register("map", map)

    pop = toolbox.population(n=mu)

    best = []
    best_score = 1e308
    last_progress = 0

    result = Dict()

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = [
            fit,
        ]
        if fit < best_score:
            best_score = fit
            best = ind

    # Begin the generational process
    for gen in range(1, ngen):
        multiprocessing.get_logger().info("%s %s %s", gen, mu, last_progress)
        offspring = algorithms.varAnd(pop, toolbox, 1.0, 1.0)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = [
                fit,
            ]

            if fit < best_score:
                best_score = fit
                best = ind
                last_progress = gen

        if gen - last_progress > stop:
            break

        # Select the next generation population
        pop = toolbox.select(pop + offspring, mu)

    result.x = best
    result.fun = best_score
    result.population = pop

    if gen < (ngen - 1):
        result.success = True
    else:
        result.success = True

    result.gen = gen

    return result


# From DEAP and modified, need to get rid of this entirely and figure out how to just build the class
# I modified this version so it does not create a global class
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
