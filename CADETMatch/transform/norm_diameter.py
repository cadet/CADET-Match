import CADETMatch.util as util
import math
from CADETMatch.abstract.transform import AbstractTransform

class DiameterTransform(AbstractTransform):
    @property
    def name(self):
        return "null"

    @property
    def count(self):
        return 1

    @property
    def count_extended(self):
        return self.count

    def transform(self):
        def trans(i):
            return i

        return [trans,]

    grad_transform = transform

    def untransform(self, seq):
        values = [math.pi * seq[0]**2/4.0,]
        headerValues = [seq[0], values[0]]
        return values, headerValues

    def grad_untransform(self, seq):
        return self.untransform(seq)[0]

    def untransform_matrix(self, matrix):
        values = numpy.zeros(matrix.shape)
        values[:,0] = math.pi * matrix[:,0]**2/4.0
        return values

    untransform_matrix_inputorder = untransform_matrix

    def setSimulation(self, sim, seq, experiment):
        values, headerValues = self.untransform(seq)

        if self.parameter.get('experiments', None) is None or experiment['name'] in self.parameter['experiments']:
            location = self.parameter['location']
            sim[location.lower()] = values[0]
        return values, headerValues

    def setupTarget(self):
        location = self.parameter['location']
        bound = -1
        comp = -1

        name = location.rsplit('/', 1)[-1]
        sensitivityOk = 1

        try:
            unit = int(location.split('/')[3].replace('unit_', ''))
        except ValueError:
            unit = ''
            sensitivityOk = 0

        return [(name, unit, comp, bound),], sensitivityOk

    def getBounds(self):
        minValue = self.parameter['min']
        maxValue = self.parameter['max']

        return [minValue,], [maxValue,]

    def getGradBounds(self):
        return [self.parameter['min'],], [self.parameter['max'],]

    def getHeaders(self):
        headers = []
        headers.append("Area")
        headers.append("Diameter")
        return headers

    def getHeadersActual(self):
        headers = []
        headers.append("Area")
        return headers

    def setBounds(self, parameter, lb, ub):
        parameter['min'] = lb[0]
        parameter['max'] = ub[0]

class NormDiameterTransform(DiameterTransform):
    @property
    def name(self):
        return "norm_diameter"

    def transform(self):
        minValue = self.parameter['min']
        maxValue = self.parameter['max']

        def trans(i):
            return (i - minValue)/(maxValue-minValue)

        return [trans,]

    grad_transform = transform

    def untransform(self, seq):
        minValue = self.parameter['min']
        maxValue = self.parameter['max']

        values_transform = [(maxValue - minValue) * seq[0] + minValue,]

        values = [math.pi * values_transform[0]**2/4.0,]
        headerValues = [values[0], values_transform[0]]
        return values, headerValues

    def grad_untransform(self, seq):
        return self.untransform(seq)[0]

    def untransform_matrix(self, matrix):
        values = self.untransform_matrix_inputorder(matrix)
        values[:,0] = math.pi * values[:,0]**2/4.0
        return values

    def untransform_matrix_inputorder(self, matrix):
        minValue = self.parameter['min']
        maxValue = self.parameter['max']

        values = (maxValue - minValue) * matrix + minValue
        return values

    def getBounds(self):
        return [0.0,], [1.0,]

    def getGradBounds(self):
        return [self.parameter['min'],], [self.parameter['max'],]

plugins = {"norm_diameter": NormDiameterTransform, "diameter": DiameterTransform}
    

