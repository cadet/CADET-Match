from abc import ABC, abstractmethod

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
        return location.split('/')[3]

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
    def setupTarget(self):
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