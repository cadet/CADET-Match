import CADETMatch.util as util
from CADETMatch.abstract.transform import AbstractTransform
import numpy

class LogTransform(AbstractTransform):
    @property
    def name(self):
        return "log"

    @property
    def count(self):
        return 1

    @property
    def count_extended(self):
        return self.count

    def transform(self):
        def trans(i):
            return numpy.log(i)

        return [trans,]

    grad_transform = transform

    def untransform(self, seq):
        values = [numpy.exp(seq[0]),]
        headerValues = values
        return values, headerValues

    def grad_untransform(self, seq):
        return self.untransform(seq)[0]

    def untransform_matrix(self, matrix):
        values = numpy.exp(matrix)
        return values

    untransform_matrix_inputorder = untransform_matrix

    def setSimulation(self, sim, seq, experiment):
        values, headerValues = self.untransform(seq)

        if self.parameter.get('experiments', None) is None or experiment['name'] in self.parameter['experiments']:
            location = self.parameter['location']
    
            try:
                comp = self.parameter['component']
                bound = self.parameter['bound']
                index = None
            except KeyError:
                index = self.parameter['index']
                bound = None

            if bound is not None:
                unit = self.getUnit(location)
                boundOffset = util.getBoundOffset(sim.root.input.model[unit])

                if comp == -1:
                    position = ()
                    sim[location.lower()] = values[0]
                else:
                    position = boundOffset[comp] + bound
                    sim[location.lower()][position] = values[0]

            if index is not None:
                sim[location.lower()][index] = values[0]
        return values, headerValues

    def setupTarget(self):
        location = self.parameter['location']
        bound = self.parameter['bound']
        comp = self.parameter['component']

        name = location.rsplit('/', 1)[-1]
        sensitivityOk = 1

        try:
            unit = int(location.split('/')[3].replace('unit_', ''))
        except ValueError:
            unit = ''
            sensitivityOk = 0

        return [(name, unit, comp, bound),], sensitivityOk

    def getBounds(self):
        minValue = numpy.log(self.parameter['min'])
        maxValue = numpy.log(self.parameter['max'])

        return [minValue,], [maxValue,]

    def getGradBounds(self):
        return [self.parameter['min'],], [self.parameter['max'],]

    def getHeaders(self):
        location = self.parameter['location']

        try:
            comp = self.parameter['component']
        except KeyError:
            comp = 'None'

        name = location.rsplit('/', 1)[-1]
        bound = self.parameter.get('bound', None)
        index = self.parameter.get('index', None)
    
        headers = []
        if bound is not None:
            headers.append("%s Comp:%s Bound:%s" % (name, comp, bound))
        if index is not None:
            headers.append("%s Comp:%s Index:%s" % (name, comp, index))
        return headers

    def getHeadersActual(self):
        return self.getHeaders()

    def setBounds(self, parameter, lb, ub):
        parameter['min'] = lb[0]
        parameter['max'] = ub[0]

class NormLogTransform(LogTransform):
    @property
    def name(self):
        return "norm_log"

    def transform(self):
        minValue = numpy.log(self.parameter['min'])
        maxValue = numpy.log(self.parameter['max'])

        def trans(i):
            return (numpy.log(i) - minValue)/(maxValue-minValue)

        return [trans,]

    grad_transform = transform

    def untransform(self, seq):
        minValue = numpy.log(self.parameter['min'])
        maxValue = numpy.log(self.parameter['max'])

        values = [(maxValue - minValue) * seq[0] + minValue,]

        values = [numpy.exp(values[0]),]
        headerValues = values
        return values, headerValues

    def grad_untransform(self, seq):
        return self.untransform(seq, self.cache, self.parameter)[0]

    def untransform_matrix(self, matrix):
        minValue = numpy.log(self.parameter['min'])
        maxValue = numpy.log(self.parameter['max'])

        temp = (maxValue - minValue) * matrix + minValue

        values = numpy.exp(temp)

        return values

    untransform_matrix_inputorder = untransform_matrix

    def getBounds(self):
        return [0.0,], [1.0,]

    def getGradBounds(self):
        return [self.parameter['min'],], [self.parameter['max'],]

plugins = {"norm_log": NormLogTransform, "log": LogTransform}