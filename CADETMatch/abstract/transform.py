from abc import ABC, abstractmethod
from CADETMatch import util


class AbstractTransform(ABC):
    def __init__(self, parameter, cache):
        self.cache = cache
        self.parameter = parameter
        super(AbstractTransform, self).__init__()

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def count(self):
        pass

    @property
    @abstractmethod
    def count_extended(self):
        pass

    def getUnit(self, location):
        return location.split("/")[3]

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def grad_transform(self):
        pass

    @abstractmethod
    def untransform(self, seq):
        pass

    @abstractmethod
    def untransform_inputorder(self, seq):
        pass

    @abstractmethod
    def grad_untransform(self, seq):
        pass

    @abstractmethod
    def untransform_matrix(self, matrix):
        pass

    @abstractmethod
    def untransform_matrix_inputorder(self, matrix):
        pass

    @abstractmethod
    def setSimulation(self, sim, seq, experiment):
        pass

    @abstractmethod
    def getBounds(self):
        pass

    @abstractmethod
    def getGradBounds(self):
        pass

    @abstractmethod
    def getHeaders(self):
        pass

    @abstractmethod
    def getHeadersActual(self):
        pass

    @abstractmethod
    def setBounds(self, parameter, lb, ub):
        pass

    def getValue(self, sim, location, bound=None, comp=None, index=None):
        if bound is not None:
            unit = self.getUnit(location)
            boundOffset = util.getBoundOffset(sim.root.input.model[unit])

            if comp == -1:
                position = ()
                return sim[location.lower()]
            else:
                position = boundOffset[comp] + bound
                return sim[location.lower()][position]

        if index is not None:
            if index == -1:
                return sim[location.lower()]
            else:
                return sim[location.lower()][index]

    def setValue(self, sim, value, location, bound=None, comp=None, index=None):
        if bound is not None:
            unit = self.getUnit(location)
            boundOffset = util.getBoundOffset(sim.root.input.model[unit])

            if comp == -1:
                position = ()
                sim[location.lower()] = value
            else:
                position = boundOffset[comp] + bound
                sim[location.lower()][position] = value

        if index is not None:
            if index == -1:
                sim[location.lower()] = value
            else:
                sim[location.lower()][index] = value
