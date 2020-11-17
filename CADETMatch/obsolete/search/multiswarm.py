import itertools
import math
import operator
import random

import numpy

name = "Multiswarm"


def run(cache, tools, creator):
    "run the parameter estimation"
    NSWARMS = 1
    NPARTICLES = 5
    NEXCESS = 3
    RCLOUD = 0.5  # 0.5 times the move severity

    # Generate the initial population
    population = [cache.toolbox.swarm(n=NPARTICLES) for _ in range(NSWARMS)]

    # Evaluate each particle
    for swarm in population:
        for part in swarm:
            fit, csv_line, results = cache.toolbox.evaluate(part)
            part.fitness.values = fit

            # Update swarm's attractors personal best and global best
            if not part.best or part.fitness > part.bestfit:
                part.best = cache.toolbox.clone(part[:])  # Get the position
                part.bestfit.values = part.fitness.values  # Get the fitness
            if not swarm.best or part.fitness > swarm.bestfit:
                swarm.best = cache.toolbox.clone(part[:])  # Get the position
                swarm.bestfit.values = part.fitness.values  # Get the fitness

    generation = 1
    while generation < 10:
        # Check for convergence
        rexcl = (BOUNDS[1] - BOUNDS[0]) / (2 * len(population) ** (1.0 / NDIM))

        not_converged = 0
        worst_swarm_idx = None
        worst_swarm = None
        for i, swarm in enumerate(population):
            # Compute the diameter of the swarm
            for p1, p2 in itertools.combinations(swarm, 2):
                d = math.sqrt(sum((x1 - x2) ** 2.0 for x1, x2 in zip(p1, p2)))
                if d > 2 * rexcl:
                    not_converged += 1
                    # Search for the worst swarm according to its global best
                    if not worst_swarm or swarm.bestfit < worst_swarm.bestfit:
                        worst_swarm_idx = i
                        worst_swarm = swarm
                    break

        # If all swarms have converged, add a swarm
        if not_converged == 0:
            population.append(cache.toolbox.swarm(n=NPARTICLES))
        # If too many swarms are roaming, remove the worst swarm
        elif not_converged > NEXCESS:
            population.pop(worst_swarm_idx)

        # Update and evaluate the swarm
        for swarm in population:
            # Check for change
            if (
                swarm.best
                and cache.toolbox.evaluate(swarm.best)[0] != swarm.bestfit.values
            ):
                # Convert particles to quantum particles
                swarm[:] = cache.toolbox.convert(
                    swarm, rcloud=RCLOUD, centre=swarm.best
                )
                swarm.best = None
                del swarm.bestfit.values

            for part in swarm:
                # Not necessary to update if it is a new swarm
                # or a swarm just converted to quantum
                if swarm.best and part.best:
                    toolbox.update(part, swarm.best)
                fit, csv_line, results = cache.toolbox.evaluate(part)
                part.fitness.values = fit

                # Update swarm's attractors personal best and global best
                if not part.best or part.fitness > part.bestfit:
                    part.best = cache.toolbox.clone(part[:])
                    part.bestfit.values = part.fitness.values
                if not swarm.best or part.fitness > swarm.bestfit:
                    swarm.best = cache.toolbox.clone(part[:])
                    swarm.bestfit.values = part.fitness.values

        # Apply exclusion
        reinit_swarms = set()
        for s1, s2 in itertools.combinations(range(len(population)), 2):
            # Swarms must have a best and not already be set to reinitialize
            if (
                population[s1].best
                and population[s2].best
                and not (s1 in reinit_swarms or s2 in reinit_swarms)
            ):
                dist = 0
                for x1, x2 in zip(population[s1].best, population[s2].best):
                    dist += (x1 - x2) ** 2.0
                dist = math.sqrt(dist)
                if dist < rexcl:
                    if population[s1].bestfit <= population[s2].bestfit:
                        reinit_swarms.add(s1)
                    else:
                        reinit_swarms.add(s2)

        # Reinitialize and evaluate swarms
        for s in reinit_swarms:
            population[s] = cache.toolbox.swarm(n=NPARTICLES)
            for part in population[s]:
                fit, csv_line, results = cache.toolbox.evaluate(part)
                part.fitness.values = fit

                # Update swarm's attractors personal best and global best
                if not part.best or part.fitness > part.bestfit:
                    part.best = cache.toolbox.clone(part[:])
                    part.bestfit.values = part.fitness.values
                if not population[s].best or part.fitness > population[s].bestfit:
                    population[s].best = cache.toolbox.clone(part[:])
                    population[s].bestfit.values = part.fitness.values
        generation += 1


def generate(pclass, pmin, pmax, smin, smax, cache):
    part = pclass(numpy.random.uniform(pmin, pmax))
    part.speed = numpy.random.uniform(smin, smax)
    return part


def convertQuantum(swarm, rcloud, centre, dist):
    dim = len(swarm[0])
    for part in swarm:
        position = [random.gauss(0, 1) for _ in range(dim)]
        distance = math.sqrt(sum(x ** 2 for x in position))

        if dist == "gaussian":
            u = abs(random.gauss(0, 1.0 / 3.0))
            part[:] = [
                (rcloud * x * u ** (1.0 / dim) / distance) + c
                for x, c in zip(position, centre)
            ]

        elif dist == "uvd":
            u = random.random()
            part[:] = [
                (rcloud * x * u ** (1.0 / dim) / distance) + c
                for x, c in zip(position, centre)
            ]

        elif dist == "nuvd":
            u = abs(random.gauss(0, 1.0 / 3.0))
            part[:] = [
                (rcloud * x * u / distance) + c for x, c in zip(position, centre)
            ]

        del part.fitness.values
        del part.bestfit.values
        part.best = None

    return swarm


def updateParticle(part, best, chi, c):
    ce1 = (c * random.uniform(0, 1) for _ in range(len(part)))
    ce2 = (c * random.uniform(0, 1) for _ in range(len(part)))
    ce1_p = map(operator.mul, ce1, map(operator.sub, best, part))
    ce2_g = map(operator.mul, ce2, map(operator.sub, part.best, part))
    a = map(
        operator.sub,
        map(operator.mul, itertools.repeat(chi), map(operator.add, ce1_p, ce2_g)),
        map(operator.mul, itertools.repeat(1 - chi), part.speed),
    )
    part.speed = list(map(operator.add, part.speed, a))
    part[:] = list(map(operator.add, part, part.speed))


def setupDEAP(
    cache, fitness, grad_fitness, grad_search, map_function, creator, base, tools
):
    "setup the DEAP variables"
    speed_min = -(numpy.array(cache.MAX_VALUE) - numpy.array(cache.MIN_VALUE)) / 2.0
    speed_max = (numpy.array(cache.MAX_VALUE) - numpy.array(cache.MIN_VALUE)) / 2.0

    creator.create("FitnessMax", base.Fitness, weights=[1.0] * cache.numGoals)
    creator.create(
        "Particle",
        list,
        fitness=creator.FitnessMax,
        speed=list,
        best=None,
        bestfit=creator.FitnessMax,
    )
    creator.create("Swarm", list, best=None, bestfit=creator.FitnessMax)

    cache.toolbox.register(
        "particle",
        generate,
        creator.Particle,
        pmin=cache.MIN_VALUE,
        pmax=cache.MAX_VALUE,
        smin=speed_min,
        smax=speed_max,
        cache=cache,
    )

    cache.toolbox.register(
        "swarm", tools.initRepeat, creator.Swarm, cache.toolbox.particle
    )

    cache.toolbox.register("update", updateParticle, chi=0.729843788, c=2.05)
    cache.toolbox.register("convert", convertQuantum, dist="nuvd")
    cache.toolbox.register("evaluate", fitness, json_path=cache.json_path)
    cache.toolbox.register("evaluate_grad", grad_fitness, json_path=cache.json_path)
    cache.toolbox.register("grad_search", grad_search)

    # cache.toolbox.register('map', map_function)
