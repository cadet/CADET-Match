from deap import base
import numpy

class Fitness2(base.Fitness):

    def dominates(self, other, obj=slice(None)):
        """Return true if each objective of *self* is not strictly worse than
        the corresponding objective of *other* and at least one objective is
        strictly better.
        :param obj: Slice indicating on which objectives the domination is
                    tested. The default value is `slice(None)`, representing
                    every objectives.
        """
        not_equal = False

        self_weight = numpy.array(self.wvalues[obj])
        other_weight = numpy.array(other.wvalues[obj])

        bools = self_weight >= other_weight
        greater_bools = self_weight > other_weight

        if numpy.all(bools) and numpy.sum(greater_bools) >= 2:
            not_equal = True
        else:
            not_equal = False
        return not_equal

class Fitness3(base.Fitness):

    def dominates(self, other, obj=slice(None)):
        """Return true if each objective of *self* is not strictly worse than
        the corresponding objective of *other* and at least one objective is
        strictly better.
        :param obj: Slice indicating on which objectives the domination is
                    tested. The default value is `slice(None)`, representing
                    every objectives.
        """
        not_equal = False

        self_weight = numpy.array(self.wvalues[obj])
        other_weight = numpy.array(other.wvalues[obj])

        bools = self_weight >= other_weight
        greater_bools = self_weight > other_weight

        if numpy.all(bools) and numpy.sum(greater_bools) >= 3:
            not_equal = True
        else:
            not_equal = False
        return not_equal