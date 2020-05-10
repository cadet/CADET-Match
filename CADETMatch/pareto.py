from deap import tools
from operator import eq
import math
import numpy

smallest = numpy.finfo(1.0).tiny

diff_step = 2.5e-2

class ParetoFront(tools.ParetoFront):
    "Modification of the pareto front in DEAP that takes cache as an argument to update to use for similar comparison"

    def __init__(self, similar=None, similar_fit = None, slice_object = None):
        if similar is None:
            similar = eq
        if similar_fit is not None:
            self.similar_fit = similar_fit
        else:
            self.similar_fit = eq
        self.slice_object = slice_object
        super().__init__(similar)

    def update(self, population):
        """Update the Pareto front hall of fame with the *population* by adding 
        the individuals from the population that are not dominated by the hall
        of fame. If any individual in the hall of fame is dominated it is
        removed.
        
        :param population: A list of individual with a fitness attribute to
                           update the hall of fame with.
        """
        new_members = []
        significant = []
        slice_object = self.slice_object
        if slice_object is None:
            slice_object = slice(None)

        for ind in population:
            is_dominated = False
            dominates_one = False
            has_twin = False
            to_remove = []
            for i, hofer in enumerate(self):    # hofer = hall of famer
                if not dominates_one and hofer.fitness.dominates(ind.fitness, obj=slice_object):
                    is_dominated = True
                    break
                elif ind.fitness.dominates(hofer.fitness, obj=slice_object):
                    dominates_one = True
                    to_remove.append(i)
                    significant.append(not self.similar_fit(ind.fitness.values, hofer.fitness.values))
                elif self.similar_fit(ind.fitness.values, hofer.fitness.values) and self.similar(ind, hofer):
                    has_twin = True
                    break
            
            for i in reversed(to_remove):       # Remove the dominated hofer
                self.remove(i)
            if not is_dominated and not has_twin:
                self.insert(ind)
                new_members.append(ind)
                significant.append(True)
        return new_members, any(significant)

    def getBestScores(self):
        weights = numpy.array(self[0].fitness.weights)
        data_meta = numpy.array([i.fitness.values for i in self])
        return numpy.max(data_meta*weights, 0)*weights

class DummyFront(tools.ParetoFront):
    "Modification of the pareto front in DEAP that takes cache as an argument to update to use for similar comparison"

    def __init__(self, similar=None):
        "This is here for API compatibility, don't do anything"
        if similar is None:
            similar = eq
        super().__init__(similar)

    def update(self, population, live_mode=True):
        "do not put anything in this front, it is just needed to maintain compatibility"
        return [], False

def similar(a, b):
    "we only need a parameter to 4 digits of accuracy so have the pareto front only keep up to 5 digits for members of the front"
    a = numpy.absolute(numpy.array(a))
    b = numpy.absolute(numpy.array(b))
    
    #used to catch division by zero
    a[a == 0.0] = smallest

    diff = numpy.abs((a-b)/a)
    return numpy.all(diff < diff_step)

def similar_fit_norm(a, b):
    "we only need a parameter to 4 digits of accuracy so have the pareto front only keep up to 5 digits for members of the front"
    a = numpy.absolute(numpy.array(a))
    b = numpy.absolute(numpy.array(b))

    #used to catch division by zero
    a[a == 0.0] = smallest

    diff = numpy.abs((a-b)/a)
    return numpy.all(diff < diff_step)

def similar_fit_sse(a, b):
    "we only need a parameter to 4 digits of accuracy so have the pareto front only keep up to 5 digits for members of the front"
    a = numpy.absolute(numpy.array(a))
    b = numpy.absolute(numpy.array(b))

    #used to catch division by zero
    a[a == 0.0] = smallest

    a = numpy.log(a)
    b = numpy.log(b)

    diff = numpy.abs((a-b)/a)
    return numpy.all(diff < diff_step)

def similar_fit_meta_norm(a, b):
    a = numpy.array(a)
    b = numpy.array(b)

    #used to catch division by zero
    a[a == 0.0] = smallest

    #SSE is in the last slot so we only want to use the first 3 meta scores
    a = a[:3]
    b = b[:3]

    diff = numpy.abs((a-b)/a)
    return numpy.all(diff < diff_step)

def similar_fit_meta_sse(a, b):
    "SSE is negative and in the last slot and the only score needed"
    a = a[-2]
    b = b[-2]

    #used to catch division by zero
    if a == 0.0:
       a = smallest

    #SSE should be handled in log scale
    a = math.log(a)
    b = math.log(b)

    diff = numpy.abs((a-b)/a)
    return numpy.all(diff < diff_step)

def similar_fit_meta_sse_split(a, b):
    "SSE is negative and in the last slot and the only score needed"
    a = numpy.abs(numpy.array(a))
    b = numpy.abs(numpy.array(b))

    #used to catch division by zero
    a[a == 0.0] = smallest

    #SSE should be handled in log scale
    a = numpy.log(a[:3])
    b = numpy.log(b[:3])

    diff = numpy.abs((a-b)/a)
    return numpy.all(diff < diff_step)

def similar_fit_meta(cache):
    if cache.allScoreNorm:
        return similar_fit_meta_norm
    elif cache.allScoreSSE:
        if cache.MultiObjectiveSSE:
            return similar_fit_meta_sse_split
        else:
            return similar_fit_meta_sse

def similar_fit(cache):
    if cache.allScoreNorm:
        return similar_fit_norm
    elif cache.allScoreSSE:
        return similar_fit_sse

def updateParetoFront(halloffame, offspring, cache):
    new_members, significant  = halloffame.update([offspring,])
    return bool(new_members), significant