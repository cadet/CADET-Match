import CADETMatch.util as util
from CADETMatch.abstract.transform import AbstractTransform

class SetTransform(AbstractTransform):
    @property
    def name(self):
        return "set_value"

    @property
    def count(self):
        return 0

    @property
    def count_extended(self):
        return 0

    def transform(self):
        return []

    grad_transform = transform

    def untransform(self, seq):
        return [], []

    def grad_untransform(self, seq):
        return self.untransform(seq)[0]

    def untransform_matrix(self, matrix):
        return None

    untransform_matrix_inputorder = untransform_matrix

    def setSimulation(self, sim, seq, experiment):
        if self.parameter.get('experiments', None) is None or experiment['name'] in self.parameter['experiments']:    
            locationFrom = self.parameter['locationFrom']
            locationTo = self.parameter['locationTo']
    
            try:
                compFrom = self.parameter['componentFrom']
                boundFrom = self.parameter['boundFrom']
                indexFrom = None
            except KeyError:
                indexFrom = self.parameter['indexFrom']
                boundFrom = None
                compFrom = None
            valueFrom = self.getValue(sim, locationFrom, bound=boundFrom, comp=compFrom, index=indexFrom)

            try:
                compTo = self.parameter['componentTo']
                boundTo = self.parameter['boundTo']
                indexTo = None
            except KeyError:
                indexTo = self.parameter['indexTo']
                boundTo = None
                compTo = None
            self.setValue(sim, valueFrom, locationTo, bound=boundTo, comp=compTo, index=indexTo)

        return [],[]

    def getBounds(self):
        return None,None

    def getGradBounds(self):
        return None, None

    def getHeaders(self):
        return []

    def getHeadersActual(self):
        return self.getHeaders()

    def setBounds(self, parameter, lb, ub):
        return None

plugins = {"set_value": SetTransform}