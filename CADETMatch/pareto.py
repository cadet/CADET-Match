from deap import tools
from operator import eq

class ParetoFront(tools.ParetoFront):
    "Modification of the pareto front in DEAP that takes cache as an argument to update to use for similar comparison"

    def __init__(self, similar=None, similar_fit = None):
        if similar is None:
            similar = eq
        if similar_fit is not None:
            self.similar_fit = similar_fit
        else:
            self.similar_fit = eq
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
        for ind in population:
            is_dominated = False
            dominates_one = False
            has_twin = False
            to_remove = []
            for i, hofer in enumerate(self):    # hofer = hall of famer
                if not dominates_one and hofer.fitness.dominates(ind.fitness):
                    is_dominated = True
                    break
                elif ind.fitness.dominates(hofer.fitness) and not self.similar_fit(ind.fitness.values, hofer.fitness.values):
                    dominates_one = True
                    to_remove.append(i)                
                elif self.similar_fit(ind.fitness.values, hofer.fitness.values) and self.similar(ind, hofer):
                    has_twin = True
                    break
            
            for i in reversed(to_remove):       # Remove the dominated hofer
                self.remove(i)
            if not is_dominated and not has_twin:
                self.insert(ind)
                new_members.append(ind)
        return new_members

class DummyFront(tools.ParetoFront):
    "Modification of the pareto front in DEAP that takes cache as an argument to update to use for similar comparison"

    def __init__(self, similar=None):
        "This is here for API compatibility, don't do anything"
        if similar is None:
            similar = eq
        super().__init__(similar)

    def update(self, population):
        "do not put anything in this front, it is just needed to maintain compatibility"
        pass

def similar(a, b):
    "we only need a parameter to 4 digits of accuracy so have the pareto front only keep up to 5 digits for members of the front"
    a = numpy.absolute(numpy.array(a))
    b = numpy.absolute(numpy.array(b))
    
    #used to catch division by zero
    a[a == 0.0] = smallest

    diff = numpy.abs((a-b)/a)
    return numpy.all(diff < 1e-2)

def similar_fit(a, b):
    "we only need a parameter to 4 digits of accuracy so have the pareto front only keep up to 5 digits for members of the front"
    a = numpy.absolute(numpy.array(a))
    b = numpy.absolute(numpy.array(b))

    #used to catch division by zero
    a[a == 0.0] = smallest

    diff = numpy.abs((a-b)/a)
    return numpy.all(diff < 1e-2)

def similar_fit_meta(a, b):
    "we only need a parameter to 4 digits of accuracy so have the pareto front only keep up to 5 digits for members of the front"
    a = numpy.absolute(numpy.array(a))
    b = numpy.absolute(numpy.array(b))

    #used to catch division by zero
    a[a == 0.0] = smallest

    #SSE is in the last slot of the scores and needs to be handled differently since it changes so rapidly compared to other scores
    a[-1] = numpy.log(a[-1])
    b[-1] = numpy.log(b[-1])

    diff = numpy.abs((a-b)/a)
    return numpy.all(diff < 1e-2)

def updateParetoFront(halloffame, offspring, cache):
    new_members  = halloffame.update([offspring,])
    return bool(new_members)