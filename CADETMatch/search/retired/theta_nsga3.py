import CADETMatch.util as util
import CADETMatch.checkpoint_algorithms as checkpoint_algorithms
import random

import numpy
from deap import tools

name = "ThetaNSGA3"

def run(cache, tools, creator):
    "run the parameter estimation"
    random.seed()
    parameters = len(cache.MIN_VALUE)

    populationSize=parameters * cache.settings['population']
    CXPB = cache.settings['crossoverRate']

    totalGenerations = parameters * cache.settings['generations']

    pop = cache.toolbox.population(n=populationSize)

    if "seeds" in cache.settings:
        seed_pop = [cache.toolbox.individual_guess([f(v) for f, v in zip(cache.settings['transform'], sublist)]) for sublist in cache.settings['seeds']]
        pop.extend(seed_pop)

    return checkpoint_algorithms.eaMuPlusLambda(pop, cache.toolbox,
                              mu=populationSize, 
                              lambda_=populationSize, 
                              cxpb=CXPB, 
                              mutpb=cache.settings['mutationRate'],
                              ngen=totalGenerations,
                              settings=cache.settings,
                              tools=tools,
                              cache=cache)

def setupDEAP(cache, fitness, grad_fitness, grad_search, map_function, creator, base, tools):
    "setup the DEAP variables"
    creator.create("FitnessMax", base.Fitness, weights=[1.0] * cache.numGoals)
    creator.create("Individual", list, typecode="d", fitness=creator.FitnessMax, strategy=None)

    cache.toolbox.register("individual", util.generateIndividual, creator.Individual,
        len(cache.MIN_VALUE), cache.MIN_VALUE, cache.MAX_VALUE, cache)
    cache.toolbox.register("population", tools.initRepeat, list, cache.toolbox.individual)

    cache.toolbox.register("individual_guess", util.initIndividual, creator.Individual, cache)

    cache.toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=5.0, low=cache.MIN_VALUE, up=cache.MAX_VALUE)

    if cache.adaptive:
        cache.toolbox.register("mutate", util.mutPolynomialBoundedAdaptive, eta=10.0, low=cache.MIN_VALUE, up=cache.MAX_VALUE, indpb=1.0/len(cache.MIN_VALUE))
    else:
        cache.toolbox.register("mutate", tools.mutPolynomialBounded, eta=2.0, low=cache.MIN_VALUE, up=cache.MAX_VALUE, indpb=1.0/len(cache.MIN_VALUE))

    cache.toolbox.register("select", sel_nsga_iii)
    cache.toolbox.register("evaluate", fitness, json_path=cache.json_path)
    cache.toolbox.register("evaluate_grad", grad_fitness, json_path=cache.json_path)
    cache.toolbox.register('grad_search', grad_search)

    cache.toolbox.register('map', map_function)

class ReferencePoint(list):
    '''A reference point exists in objective space an has a set of individuals
    associated to it.'''
    def __init__(self, *args):
        list.__init__(self, *args)
        self.associations_count = 0
        self.associations = []

def generate_reference_points(num_objs, num_divisions_per_obj=4):
    '''Generates reference points for NSGA-III selection. This code is based on
    `jMetal NSGA-III implementation <https://github.com/jMetal/jMetal>`_.
    '''
    #What are each of these variables and what should I do with them?
    # K = number of reference points
    # m = number of objects
    # s = sum of each reference points should be 1
    # lambda_jk = j,k entries of a larger matrix of reference points
    #
    #for j in range(1, K):
    #    s = 0
    #    for k in range(1, m):
    #        if k < m:
    #            r = random(0,1)
    #            lambda_jk = (1-s) * (1-r**(1/(m-k)))
    #            s = s + lambda_jk
    #        else:
    #            lambda_jk = 1-s


def find_ideal_point(individuals):
    'Finds the ideal point from a set individuals.'
    current_ideal = [numpy.infty] * len(individuals[0].fitness.values)
    for ind in individuals:
        # Use wvalues to accomodate for maximization and minimization problems.
        current_ideal = numpy.minimum(current_ideal,
                                   numpy.multiply(ind.fitness.wvalues, -1))
    return current_ideal

def find_extreme_points(individuals):
    'Finds the individuals with extreme values for each objective function.'
    return [sorted(individuals, key=lambda ind: ind.fitness.wvalues[o] * -1)[-1]
            for o in range(len(individuals[0].fitness.values))]

def construct_hyperplane(individuals, extreme_points):
    'Calculates the axis intersects for a set of individuals and its extremes.'
    def has_duplicate_individuals(individuals):
        for i in range(len(individuals)):
            for j in range(i+1, len(individuals)):
                if individuals[i].fitness.values == individuals[j].fitness.values:
                    return True
        return False

    num_objs = len(individuals[0].fitness.values)

    if has_duplicate_individuals(extreme_points):
        intercepts = [extreme_points[m].fitness.values[m] for m in range(num_objs)]
    else:
        b = numpy.ones(num_objs)
        A = [point.fitness.values for point in extreme_points]
        x = numpy.linalg.solve(A, b)
        intercepts = 1/x
    return intercepts

def normalize_objective(individual, m, intercepts, ideal_point, epsilon=1e-20):
    'Normalizes an objective.'
    # Numeric trick present in JMetal implementation.
    if numpy.abs(intercepts[m]-ideal_point[m] > epsilon):
        return individual.fitness.values[m] / (intercepts[m]-ideal_point[m])
    else:
        return individual.fitness.values[m] / epsilon

def normalize_objectives(individuals, intercepts, ideal_point):
    '''Normalizes individuals using the hyperplane defined by the intercepts as
    reference. Corresponds to Algorithm 2 of Deb & Jain (2014).'''
    num_objs = len(individuals[0].fitness.values)

    for ind in individuals:
        ind.fitness.normalized_values = list([normalize_objective(ind, m,
                                                                  intercepts, ideal_point)
                                                                  for m in range(num_objs)])
    return individuals

def perpendicular_distance(direction, point):
    k = numpy.dot(direction, point) / numpy.sum(numpy.power(direction, 2))
    d = numpy.sum(numpy.power(numpy.subtract(numpy.multiply(direction, [k] * len(direction)), point) , 2))
    return numpy.sqrt(d)

def perpendicular_distances(direction, points):
    #This function is also slow but I am not sure what else to do with it
    #Need to test with numpy connected to MKL
    t = numpy.dot(points,direction)/numpy.linalg.norm(direction)**2
    d = numpy.linalg.norm(points - numpy.outer(t, direction), axis=1)
    return d

def associate(individuals, reference_points):
    '''Associates individuals to reference points and calculates niche number.
    Corresponds to Algorithm 3 of Deb & Jain (2014).'''
    pareto_fronts = tools.sortLogNondominated(individuals, len(individuals))
    num_objs = len(individuals[0].fitness.values)

    for ind in individuals:
        dists = perpendicular_distances(ind.fitness.normalized_values, reference_points)
        best_idx = numpy.argmin(dists)
        best_rp = reference_points[best_idx]
        best_dist = dists[best_idx]

        ind.reference_point = best_rp
        ind.ref_point_distance = best_dist
        best_rp.associations_count += 1 # update de niche number
        best_rp.associations += [ind]

def niching_select(individuals, k):
    '''Secondary niched selection based on reference points. Corresponds to
    steps 13-17 of Algorithm 1 and to Algorithm 4.'''
    if len(individuals) == k:
        return individuals

    #individuals = copy.deepcopy(individuals)

    ideal_point = find_ideal_point(individuals)
    extremes = find_extreme_points(individuals)
    intercepts = construct_hyperplane(individuals, extremes)
    normalize_objectives(individuals, intercepts, ideal_point)

    reference_points = generate_reference_points(len(individuals[0].fitness.values))

    ref = numpy.array(reference_points)

    associate(individuals, reference_points)

    res = []
    while len(res) < k:
        #This can still be made more efficient by updating the associations count when it changes during iteration
        #instead of rebuilding the system. Right now this takes no time at all anymore when profiling
        #unless it takes more time in higher dimensions or larger problems there is no point in further
        #optimization here

        rp_associations_count = numpy.fromiter((rp.associations_count for rp in reference_points), int, len(reference_points))

        min_assoc_rp = numpy.min(rp_associations_count[numpy.nonzero(rp_associations_count)])

        idxs = numpy.where(rp_associations_count == min_assoc_rp)

        chosen_rp = reference_points[random.choice(idxs[0])]

        associated_inds = chosen_rp.associations
        if chosen_rp.associations:
            if chosen_rp.associations_count == 0:
                sel = min(chosen_rp.associations, key=lambda ind: ind.ref_point_distance)
            else:
                sel = chosen_rp.associations[random.randint(0, len(chosen_rp.associations)-1)]
            res += [sel]
            chosen_rp.associations.remove(sel)
            chosen_rp.associations_count += 1
            individuals.remove(sel)
        else:
            reference_points.remove(chosen_rp)
    return res

def sel_nsga_iii(individuals, k):
    '''Implements NSGA-III selection as described in
    Deb, K., & Jain, H. (2014). An Evolutionary Many-Objective Optimization
    Algorithm Using Reference-Point-Based Nondominated Sorting Approach,
    Part I: Solving Problems With Box Constraints. IEEE Transactions on
    Evolutionary Computation, 18(4), 577–601. doi:10.1109/TEVC.2013.2281535.
    '''
    assert len(individuals) >= k

    if len(individuals) == k:
        return individuals

    # Algorithm 1 steps 4--8
    fronts = tools.sortLogNondominated(individuals, len(individuals))

    limit = 0
    res = []
    for f, front in enumerate(fronts):
        res += front
        if len(res) > k:
            limit = f
            break
    # Algorithm 1 steps
    selection = []
    if limit > 0:
        for f in range(limit):
            selection += fronts[f]

    # complete selected inividuals using the referece point based approach
    selection += niching_select(fronts[limit], k - len(selection))
    return selection