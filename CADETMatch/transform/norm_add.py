import CADETMatch.util as util
from CADETMatch.abstract.transform import AbstractTransform

class NormAddTransform(AbstractTransform):
    @property
    def name(self):
        return "norm_add"

    @property
    def count(self):
        if self.parameter['min'] != self.parameter['max']:
            return 1
        else:
            return 0

    @property
    def count_extended(self):
        return self.count

    def transform(self):
        if self.count:
            minValue = self.parameter['min']
            maxValue = self.parameter['max']

            def trans(i):
                return (i - minValue)/(maxValue-minValue)

            return [trans,]
        else:
            return []

    grad_transform = transform

    def untransform(self, seq):
        if self.count:
            minValue = self.parameter['min']
            maxValue = self.parameter['max']

            values = [(maxValue - minValue) * seq[0] + minValue,]
            headerValues = values
            return values, headerValues
        else:
            return [],[]

    def grad_untransform(self, seq):
        return self.untransform(seq)[0]

    def untransform_matrix(self, matrix):
        if self.count:
            minValue = self.parameter['min']
            maxValue = self.parameter['max']

            values = (maxValue - minValue) * matrix + minValue
            return values
        else:
            return None

    untransform_matrix_inputorder = untransform_matrix

    def setSimulation(self, sim, seq, experiment):
        values, headerValues = self.untransform(seq)

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

            if self.count:
                temp = values[0]
            else:
                temp = self.parameter['min']
            
            self.setValue(sim, valueFrom + temp, locationTo, bound=boundTo, comp=compTo, index=indexTo)

        if self.count:
            return values, headerValues
        else:
            return [], []

    def getBounds(self):
        if self.count:
            return [0.0,], [1.0,]
        else:
            return None, None

    def getGradBounds(self):
        if self.count:
            return [self.parameter['min'],], [self.parameter['max'],]
        else:
            return None, None

    def getHeaders(self):
        if self.count:
            location = self.parameter['locationTo']

            try:
                comp = self.parameter['componentTo']
            except KeyError:
                comp = 'None'

            name = location.rsplit('/', 1)[-1]
            bound = self.parameter.get('boundTo', None)
            index = self.parameter.get('indexTo', None)
    
            headers = []
            if bound is not None:
                headers.append("%s Comp:%s Bound:%s" % (name, comp, bound))
            elif index is not None:
                headers.append("%s Comp:%s Index:%s" % (name, comp, index))
            return headers
        else:
            return []

    def getHeadersActual(self):
        return self.getHeaders()

    def setBounds(self, parameter, lb, ub):
        parameter['min'] = lb[0]
        parameter['max'] = ub[0]

plugins = {"norm_add": NormAddTransform}    
