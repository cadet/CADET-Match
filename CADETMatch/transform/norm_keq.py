import CADETMatch.util as util
import numpy
from CADETMatch.abstract.transform import AbstractTransform

class KeqTransform(AbstractTransform):
    @property
    def name(self):
        return "keq"

    @property
    def count(self):
        return 2

    @property
    def count_extended(self):
        return 3

    def transform(self):
        def trans_a(i):
            return numpy.log(i)

        def trans_b(i):
            return numpy.log(i)

        return [trans_a, trans_b]

    grad_transform = transform

    def untransform(self, seq):
        values = [numpy.exp(seq[0]), numpy.exp(seq[0])/(numpy.exp(seq[1]))]
        headerValues = [values[0], values[1], values[0]/values[1]]
        return values, headerValues

    def grad_untransform(self, seq):
        return self.untransform(seq)[0]

    def untransform_matrix(self, matrix):
        values = self.untransform_matrix_inputorder(matrix)
        values[:,1] = values[:,0] / values[:,1]
        return values

    def untransform_matrix_inputorder(self, matrix):
        values = numpy.exp(matrix)
        return values

    def setSimulation(self, sim, seq, experiment):
        values, headerValues = self.untransform(seq)

        if self.parameter.get('experiments', None) is None or experiment['name'] in self.parameter['experiments']:
            location = self.parameter['location']
    
            comp = self.parameter['component']
            bound = self.parameter['bound']
    
            unit = self.getUnit(location[0])
            boundOffset = util.getBoundOffset(sim.root.input.model[unit])

            position = boundOffset[comp] + bound
            sim[location[0].lower()][position] = values[0]
            sim[location[1].lower()][position] = values[1]

        return values, headerValues

    def setupTarget(self):
        location = self.parameter['location']
        bound = self.parameter['bound']
        comp = self.parameter['component']

        sensitivityOk = 1
        nameKA = location[0].rsplit('/', 1)[-1]
        nameKD = location[1].rsplit('/', 1)[-1]
        unit = int(location[0].split('/')[3].replace('unit_', ''))

        return [(nameKA, unit, comp, bound), (nameKD, unit, comp, bound)], sensitivityOk

    def getBounds(self):
        minKA = self.parameter['minKA']
        maxKA = self.parameter['maxKA']
        minKEQ = self.parameter['minKEQ']
        maxKEQ = self.parameter['maxKEQ']

        minValues = list(numpy.log([minKA, minKEQ]))
        maxValues = list(numpy.log([maxKA, maxKEQ]))

        return minValues, maxValues

    def getGradBounds(self):
        return [self.parameter['min'],], [self.parameter['max'],]

    def getHeaders(self):
        location = self.parameter['location']
        nameKA = location[0].rsplit('/', 1)[-1]
        nameKD = location[1].rsplit('/', 1)[-1]
        bound = self.parameter['bound']
        comp = self.parameter['component']
    
        headers = []
        headers.append("%s Comp:%s Bound:%s" % (nameKA, comp, bound))
        headers.append("%s Comp:%s Bound:%s" % (nameKD, comp, bound))
        headers.append("%s/%s Comp:%s Bound:%s" % (nameKA, nameKD, comp, bound))
        return headers

    def getHeadersActual(self):
        location = self.parameter['location']
        nameKA = location[0].rsplit('/', 1)[-1]
        nameKD = location[1].rsplit('/', 1)[-1]
        bound = self.parameter['bound']
        comp = self.parameter['component']
    
        headers = []
        headers.append("%s Comp:%s Bound:%s" % (nameKA, comp, bound))
        headers.append("%s/%s Comp:%s Bound:%s" % (nameKA, nameKD, comp, bound))
        return headers

    def setBounds(self, parameter, lb, ub):
        parameter['minKA'] = lb[0]
        parameter['maxKA'] = ub[0]
        parameter['minKEQ'] = lb[2]
        parameter['maxKEQ'] = ub[2]

class NormKeqTransform(KeqTransform):
    @property
    def name(self):
        return "norm_keq"

    def transform(self):
        minKA = numpy.log(self.parameter['minKA'])
        maxKA = numpy.log(self.parameter['maxKA'])
        minKEQ = numpy.log(self.parameter['minKEQ'])
        maxKEQ = numpy.log(self.parameter['maxKEQ'])

        def trans_a(i):
            return (numpy.log(i) - minKA)/(maxKA-minKA)

        def trans_b(i):
            return (numpy.log(i) - minKEQ)/(maxKEQ-minKEQ)

        return [trans_a, trans_b]

    grad_transform = transform

    def untransform(self, seq):
        minKA = self.parameter['minKA']
        maxKA = self.parameter['maxKA']
        minKEQ = self.parameter['minKEQ']
        maxKEQ = self.parameter['maxKEQ']

        minValues = numpy.log([minKA, minKEQ])
        maxValues = numpy.log([maxKA, maxKEQ])

        values = numpy.array(seq)

        values = (maxValues - minValues) * values + minValues

        values = [numpy.exp(values[0]), numpy.exp(values[0])/(numpy.exp(values[1]))]
        headerValues = [values[0], values[1], values[0]/values[1]]
        return values, headerValues

    def grad_untransform(self, seq):
        return self.untransform(seq)[0]

    def untransform_matrix(self, matrix):
        values = self.untransform_matrix_inputorder(matrix)
        values[:,1] = values[:,0] / values[:,1]
        return values

    def untransform_matrix_inputorder(self, matrix):
        minKA = self.parameter['minKA']
        maxKA = self.parameter['maxKA']
        minKEQ = self.parameter['minKEQ']
        maxKEQ = self.parameter['maxKEQ']

        minValues = numpy.log([minKA, minKEQ])
        maxValues = numpy.log([maxKA, maxKEQ])
    
        values = (maxValues - minValues) * matrix + minValues

        values = numpy.exp(values)

        return values

    def getBounds(self):
        return [0.0, 0.0], [1.0, 1.0]

    def getGradBounds(self):
        return [self.parameter['min'],], [self.parameter['max'],]

plugins = {"keq": KeqTransform, "norm_keq": NormKeqTransform}