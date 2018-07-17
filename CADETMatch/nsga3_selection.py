#This file is from https://raw.githubusercontent.com/lmarti/nsgaiii/master/nsgaiii/selection.py


#    This file is part of nsgaiii, a Python implementation of NSGA-III.
#
#    nsgaiii is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    nsgaiii is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with nsgaiii. If not, see <http://www.gnu.org/licenses/>.
#
#    by Luis Marti (IC/UFF) http://lmarti.com

from __future__ import division # making it work with Python 2.x
import copy
import random
import numpy as np
from deap import tools

import SALib.sample.sobol_sequence

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

    sobol = SALib.sample.sobol_sequence.sample(100 * num_objs, num_objs)
    data = np.apply_along_axis(list, 1, sobol)
    data = list(map(ReferencePoint, data))
    return data
    #This entire function needs to be completely recoded, in high dimensions in it is major time sink
    # 287/622s. This should probably be done once and stored instead of regenerated at every iteration
    #def gen_refs_recursive(work_point, num_objs, left, total, depth):
    #    if depth == num_objs - 1:
    #        work_point[depth] = left/total
    #        ref = ReferencePoint(copy.deepcopy(work_point))
    #        return [ref]
    #    else:
    #        res = []
    #        for i in range(left):
    #            work_point[depth] = i/total
    #            res = res + gen_refs_recursive(work_point, num_objs, left-i, total, depth+1)
    #        return res
    #return gen_refs_recursive([0]*num_objs, num_objs, num_objs*num_divisions_per_obj,
    #                          num_objs*num_divisions_per_obj, 0)

def find_ideal_point(individuals):
    'Finds the ideal point from a set individuals.'
    current_ideal = [np.infty] * len(individuals[0].fitness.values)
    for ind in individuals:
        # Use wvalues to accomodate for maximization and minimization problems.
        current_ideal = np.minimum(current_ideal,
                                   np.multiply(ind.fitness.wvalues, -1))
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
        b = np.ones(num_objs)
        A = [point.fitness.values for point in extreme_points]
        x = np.linalg.solve(A, b)
        intercepts = 1/x
    return intercepts

def normalize_objective(individual, m, intercepts, ideal_point, epsilon=1e-20):
    'Normalizes an objective.'
    # Numeric trick present in JMetal implementation.
    if np.abs(intercepts[m]-ideal_point[m] > epsilon):
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
    k = np.dot(direction, point) / np.sum(np.power(direction, 2))
    d = np.sum(np.power(np.subtract(np.multiply(direction, [k] * len(direction)), point) , 2))
    return np.sqrt(d)

def perpendicular_distances(direction, points):
    #This function is also slow but I am not sure what else to do with it
    #Need to test with numpy connected to MKL
    t = np.dot(points,direction)/np.linalg.norm(direction)**2
    d = np.linalg.norm(points - np.outer(t, direction), axis=1)
    return d

def associate(individuals, reference_points):
    '''Associates individuals to reference points and calculates niche number.
    Corresponds to Algorithm 3 of Deb & Jain (2014).'''
    pareto_fronts = tools.sortLogNondominated(individuals, len(individuals))
    num_objs = len(individuals[0].fitness.values)

    for ind in individuals:
        dists = perpendicular_distances(ind.fitness.normalized_values, reference_points)
        best_idx = np.argmin(dists)
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
    #print(len(reference_points))

    ref = np.array(reference_points)
    #print(ref.shape)

    #print(set(ref[:,0]))
    #print(set(ref[:,1]))
    #print(set(ref[:,2]))
    #print(set(ref[:,3]))

    associate(individuals, reference_points)

    res = []
    while len(res) < k:
        #This can still be made more efficient by updating the associations count when it changes during iteration
        #instead of rebuilding the system. Right now this takes no time at all anymore when profiling
        #unless it takes more time in higher dimensions or larger problems there is no point in further
        #optimization here

        rp_associations_count = np.fromiter((rp.associations_count for rp in reference_points), int, len(reference_points))

        min_assoc_rp = np.min(rp_associations_count[np.nonzero(rp_associations_count)])

        idxs = np.where(rp_associations_count == min_assoc_rp)

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
    
    #These attributes are no longer needed but make the checkpoints huge        
    for i in res:
        i.reference_point = None
        i.ref_point_distance = None

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

__all__ = ["sel_nsga_iii"]
