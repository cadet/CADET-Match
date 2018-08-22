from deap import tools
from operator import eq

class ParetoFront(tools.ParetoFront):
    "Modification of the pareto front in DEAP that takes cache as an argument to update to use for similar comparison"

    def __init__(self, similar=None):
        if similar is None:
            similar = eq
        super().__init__(similar)

    def update(self, population, cache):
        """Update the Pareto front hall of fame with the *population* by adding 
        the individuals from the population that are not dominated by the hall
        of fame. If any individual in the hall of fame is dominated it is
        removed.
        
        :param population: A list of individual with a fitness attribute to
                           update the hall of fame with.
        """
        for ind in population:
            is_dominated = False
            dominates_one = False
            has_twin = False
            to_remove = []
            for i, hofer in enumerate(self):    # hofer = hall of famer
                if not dominates_one and hofer.fitness.dominates(ind.fitness):
                    is_dominated = True
                    break
                elif ind.fitness.dominates(hofer.fitness):
                    dominates_one = True
                    to_remove.append(i)
                elif ind.fitness == hofer.fitness and self.similar(ind, hofer, cache):
                    has_twin = True
                    break
            
            for i in reversed(to_remove):       # Remove the dominated hofer
                self.remove(i)
            if not is_dominated and not has_twin:
                self.insert(ind)

class DummyFront(tools.ParetoFront):
    "Modification of the pareto front in DEAP that takes cache as an argument to update to use for similar comparison"

    def __init__(self, similar=None):
        "This is here for API compatibility, don't do anything"
        if similar is None:
            similar = eq
        super().__init__(similar)

    def update(self, population, cache):
        "do not put anything in this front, it is just needed to maintain compatibility"
        pass