import numpy
import array
import attr
from typing import Iterable

def seq2array(x: Iterable[float]) -> array.array:
    return array.array("d", x)

class SetterProperty():
    def __init__(self, func):
        self.func = func
        self.__doc__ = func.__doc__
    def __set__(self, obj, value):
        return self.func(obj, value)

@attr.s
class Fitness:
    "designed for DEAP compatibility but only minimization, clean and array.array"
    _values = attr.ib(converter=seq2array)
    
    @property
    def wvalues(self):
        "wvalues assumes maximization for some reason to flip the sign"
        return tuple([-1*i for i in  self._values])

    @property
    def values(self):
        return self._values
    
    @values.setter
    def values(self, value):
        self._values = seq2array(value)

    @values.deleter
    def values(self):
        self._values = ()
    
    @property
    def valid(self):
        return len(self._values) != 0
    
    def __array__(self, dtype=None):
        return numpy.frombuffer(self._values, dtype=dtype)
    
    def dominates(self, other, obj=slice(None)):
        """Return true if each objective of *self* is not strictly worse than
        the corresponding objective of *other* and at least one objective is
        strictly better.
        :param obj: Slice indicating on which objectives the domination is
                    tested. The default value is `slice(None)`, representing
                    every objectives.
        """
        dominate = False
        for self_value, other_value in zip(self.values[obj], other.values[obj]):
            if self_value < other_value:
                dominate = True
            elif self_value > other_value:
                return False
        return dominate
    
@attr.s
class Individual:
    value = attr.ib(converter=seq2array)
    fitness = attr.ib(converter=Fitness, default=attr.Factory(list))
    best = attr.ib(default=None)
    csv_line = attr.ib(default='')
    
    @property
    def valid(self):
        return self.fitness.valid
    
    def __array__(self, dtype=None):
        return numpy.frombuffer(self.value, dtype=dtype)

    def __iter__(self):
        return iter(self.value)

    def __len__(self):
        return len(self.value)

    def __getitem__(self, index):
        return self.value[index]

    def __setitem__(self, index, value):
        self.value[index] = value