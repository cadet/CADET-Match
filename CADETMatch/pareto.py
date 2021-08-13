import copy
from operator import eq
import hashlib
import CADETMatch.pop as pop
import bisect

import numpy
class ParetoFront:
    "Modification of the pareto front in DEAP that takes cache as an argument to update to use for similar comparison"

    def __init__(self, dimensions, similar=None, similar_fit=None, slice_object=slice(None)):
        if similar is None:
            similar = eq
        
        if similar_fit is not None:
            self.similar_fit = similar_fit
        else:
            self.similar_fit = eq
        
        self.slice_object = slice_object
        self.keys = list()
        self.items = list()
        self.similar = similar

        #due to the crowding measures if solutions are very close together it is possible for the very
        #best solution for one of the goals to not be stored so they should be kept separately and
        #merged in
        self.dimensions = dimensions
        self.best_keys = [1e308]*dimensions
        self.best_items = [pop.Individual([0], [1e308]*dimensions)]*dimensions

    def insert(self, item):
        """Insert a new item in sorted order"""
        item = copy.deepcopy(item)
        i = bisect.bisect(self.keys, item.fitness)
        self.items.insert(i, item)
        self.keys.insert(i, item.fitness)

    def remove(self, index):
        del self.keys[index]
        del self.items[index]

    def clear(self):
        del self.items[:]
        del self.keys[:]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def __iter__(self):
        return iter(self.items)

    def __reversed__(self):
        return reversed(self.items)

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
        pareto_length = len(self)

        for ind in population:
            is_dominated = False
            dominates_one = False
            has_twin = False
            to_remove = []
            
            for idx in range(len(ind.fitness.values)):
                if ind.fitness.values[idx] < self.best_keys[idx]:
                    self.best_keys[idx] = ind.fitness.values[idx]
                    self.best_items[idx] = copy.deepcopy(ind)

            for i, hofer in enumerate(self):  # hofer = hall of famer
                if not dominates_one and hofer.fitness.dominates(
                    ind.fitness, obj=slice_object
                ):
                    is_dominated = True
                    break
                elif ind.fitness.dominates(hofer.fitness, obj=slice_object):
                    dominates_one = True
                    to_remove.append(i)
                    significant.append(
                        not self.similar_fit(ind.fitness.values, hofer.fitness.values)
                    )
                elif self.similar_fit(
                    ind.fitness.values, hofer.fitness.values
                ) and self.similar(ind.value, hofer.value):
                    has_twin = True
                    break

            for i in reversed(to_remove):  # Remove the dominated hofer
                self.remove(i)
            if not is_dominated and not has_twin:
                # if the pareto front is empty or a new item is added to the pareto front that is progress
                # however if there is only a single objective and it does not significantly dominate then
                # don't count that as significant progress
                if pareto_length == 0:
                    significant.append(True)
                elif self.dimensions > 1:
                    if len(significant) == 0 or (len(significant) and any(significant)):
                        significant.append(True)

                self.insert(ind)
                new_members.append(ind)

        return new_members, any(significant)

    def hashes(self):
        hashes = {
            hashlib.md5(str(list(individual.value)).encode("utf-8", "ignore")).hexdigest()
            for individual in self.items
        }
        hashes_best = {
            hashlib.md5(str(list(individual.value)).encode("utf-8", "ignore")).hexdigest()
            for individual in self.best_items
        }
        all_hashes = hashes | hashes_best
        return all_hashes 

    def getBestScores(self):
        items = [i.fitness.values for i in self.items]
        items.extend([i.fitness.values for i in self.best_items])
        data = numpy.array(items)
        return numpy.min(data, 0)

class DummyFront(ParetoFront):
    "Modification of the pareto front in DEAP that takes cache as an argument to update to use for similar comparison"

    def __init__(self, dimensions=None, similar=None):
        "This is here for API compatibility, don't do anything"
        self.items = []
        self.best_items = []
        return None

    def update(self, population):
        "do not put anything in this front, it is just needed to maintain compatibility"
        return [], False


def similar(a, b):
    "for minimization the rtol needs to be fairly high otherwise the pareto front contains too many entries"
    a = numpy.frombuffer(a)
    b = numpy.frombuffer(b)
    return numpy.allclose(a,b, rtol=1e-1)


def similar_fit_meta_split(a, b):
    a = numpy.frombuffer(a)
    b = numpy.frombuffer(b)
    return numpy.allclose(a[:3],b[:3], rtol=1e-1)


def similar_fit_meta_sse(a, b):
    "SSE is negative and in the last slot and the only score needed"
    a = numpy.frombuffer(a)
    b = numpy.frombuffer(b)
    return numpy.allclose(a[-2],b[-2], rtol=1e-1)


def similar_fit_meta(cache):
    if cache.allScoreSSE and not cache.MultiObjectiveSSE:
        return similar_fit_meta_sse
    else:
        return similar_fit_meta_split


def similar_fit(cache):
    return similar


def updateParetoFront(halloffame, offspring):
    new_members, significant = halloffame.update(
        [
            offspring,
        ]
    )
    return bool(new_members), significant

def getBestScoresMonkey(self):
    items = [i.fitness.values for i in self.items]
    data = numpy.array(items)
    return numpy.min(data, 0)