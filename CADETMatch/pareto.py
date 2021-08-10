import copy
import math
import multiprocessing
from operator import eq

import numpy
from deap import tools

smallest = numpy.finfo(1.0).tiny

diff_step = 1e-1


class ParetoFront(tools.ParetoFront):
    "Modification of the pareto front in DEAP that takes cache as an argument to update to use for similar comparison"

    def __init__(self, similar=None, similar_fit=None, slice_object=None):
        if similar is None:
            similar = eq
        if similar_fit is not None:
            self.similar_fit = similar_fit
        else:
            self.similar_fit = eq
        self.slice_object = slice_object
        super().__init__(similar)

    def update(self, population, numGoals):
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
        if slice_object is None:
            slice_object = slice(None)

        for ind in population:
            is_dominated = False
            dominates_one = False
            has_twin = False
            to_remove = []
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
                ) and self.similar(ind, hofer):
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
                elif numGoals > 1:
                    if len(significant) == 0 or (len(significant) and any(significant)):
                        significant.append(True)

                self.insert(ind)
                new_members.append(ind)

        return new_members, any(significant)

    def getBestScores(self):
        weights = numpy.array(self[0].fitness.weights)
        data_meta = numpy.array([i.fitness.values for i in self])
        return numpy.max(data_meta * weights, 0) * weights


class ParetoFrontMeta(ParetoFront):
    def __init__(self, similar=None, similar_fit=None, slice_object=None):
        super().__init__(similar, similar_fit, slice_object)
        self.best_sse = numpy.inf
        self.best_sse_ind = None
        self.best_rmse = numpy.inf
        self.best_rmse_ind = None

    def update(self, population, numGoals):
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
        if slice_object is None:
            slice_object = slice(None)

        for ind in population:
            is_dominated = False
            dominates_one = False
            has_twin = False
            to_remove = []

            ind_sse = ind.fitness.values[-2]
            ind_rmse = ind.fitness.values[-1]

            new_best_sse_found = False
            new_best_rmse_found = False

            if ind_sse < self.best_sse:
                new_best_sse = ind_sse
                new_best_sse_ind = copy.deepcopy(ind)
                new_best_sse_ind.best = True
                new_best_sse_found = True

            if ind_rmse < self.best_rmse:
                new_best_rmse = ind_rmse
                new_best_rmse_ind = copy.deepcopy(ind)
                new_best_rmse_ind.best = True
                new_best_rmse_found = True

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
                ) and self.similar(ind, hofer):
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
                elif numGoals > 1:
                    if len(significant) == 0 or (len(significant) and any(significant)):
                        significant.append(True)

                self.insert(ind)
                new_members.append(ind)

            # add sse and rmse individuals if they are not stored already
            foundSSE = False
            foundRMSE = False

            if new_best_rmse_found or new_best_sse_found:
                # remove old best
                to_remove = []
                for idx, ind in enumerate(self):
                    if ind.best:
                        removeSSE = (
                            new_best_sse_found
                            and ind.fitness.values == self.best_sse_ind.fitness.values
                        )
                        removeRMSE = (
                            new_best_rmse_found
                            and ind.fitness.values == self.best_rmse_ind.fitness.values
                        )
                        if removeSSE or removeRMSE:
                            to_remove.append(idx)

                for idx in reversed(to_remove):  # Remove the dominated hofer
                    self.remove(idx)

                if new_best_sse_found:
                    self.best_sse_ind = new_best_sse_ind
                    self.best_sse = new_best_sse

                if new_best_rmse_found:
                    self.best_rmse_ind = new_best_rmse_ind
                    self.best_rmse = new_best_rmse

            sameInd = (
                self.best_sse_ind.fitness.values == self.best_rmse_ind.fitness.values
            )

            for ind in self:
                if ind.fitness.values == self.best_sse_ind.fitness.values:
                    foundSSE = True
                if ind.fitness.values == self.best_rmse_ind.fitness.values:
                    foundRMSE = True
                if foundSSE and foundRMSE:
                    break
            if not foundSSE and sameInd:
                self.insert(self.best_sse_ind)
                new_members.append(ind)

            if not foundSSE and not sameInd:
                self.insert(self.best_sse_ind)
                new_members.append(ind)
            if not foundRMSE and not sameInd:
                self.insert(self.best_rmse_ind)
                new_members.append(ind)

        return new_members, any(significant)


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

    # used to catch division by zero
    a[a == 0.0] = smallest

    diff = numpy.abs((a - b) / a)
    return numpy.all(diff < diff_step)


def similar_fit_norm(a, b):
    "we only need a parameter to 4 digits of accuracy so have the pareto front only keep up to 5 digits for members of the front"
    a = numpy.absolute(numpy.array(a))
    b = numpy.absolute(numpy.array(b))

    # used to catch division by zero
    a[a == 0.0] = smallest

    a = numpy.log(a)
    b = numpy.log(b)

    diff = numpy.abs((a - b) / a)
    return numpy.all(diff < diff_step)


def similar_fit_meta_split(a, b):
    a = numpy.array(a)
    b = numpy.array(b)

    # used to catch division by zero
    a[a == 0.0] = smallest

    # SSE is in the last slot so we only want to use the first 3 meta scores
    a = numpy.log(a[:3])
    b = numpy.log(b[:3])

    diff = numpy.abs((a - b) / a)
    return numpy.all(diff < diff_step)


def similar_fit_meta_sse(a, b):
    "SSE is negative and in the last slot and the only score needed"
    a = a[-2]
    b = b[-2]

    # used to catch division by zero
    if a == 0.0:
        a = smallest

    # SSE should be handled in log scale
    a = math.log(a)
    b = math.log(b)

    diff = numpy.abs((a - b) / a)
    return numpy.all(diff < diff_step)


def similar_fit_meta(cache):
    if cache.allScoreSSE and not cache.MultiObjectiveSSE:
        return similar_fit_meta_sse
    else:
        return similar_fit_meta_split


def similar_fit(cache):
    return similar_fit_norm


def updateParetoFront(halloffame, offspring, cache):
    new_members, significant = halloffame.update(
        [
            offspring,
        ],
        cache.numGoals,
    )
    return bool(new_members), significant
