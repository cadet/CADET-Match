import CADETMatch.util as util
import numpy
from CADETMatch.abstract.transform import AbstractTransform

class VolumeLengthTransform(AbstractTransform):
    @property
    def name(self):
        return "volume_length"

    @property
    def count(self):
        return 2

    @property
    def count_extended(self):
        return 3

    def transform(self):
        def trans_volume(i):
            return i

        def trans_length(i):
            return i

        return [trans_volume, trans_length]

    grad_transform = transform

    def untransform(self, seq):
        values = [seq[0]/seq[1], seq[1]]
        headerValues = [values[0], values[1], values[0]*values[1]]
        return values, headerValues

    def grad_untransform(self, seq):
        return self.untransform(seq)[0]

    def untransform_matrix(self, matrix):
        values = self.untransform_matrix_inputorder(matrix)
        values[:,0] = values[:,0] / values[:,1]
        return values

    def untransform_matrix_inputorder(self, matrix):
        values = numpy.zeros(matrix.shape)
        values[:,0] = matrix[:,0]
        values[:,1] = matrix[:,1]
        return values

    def setSimulation(self, sim, seq, experiment):
        values, headerValues = self.untransform(seq)
    
        if self.parameter.get('experiments', None) is None or experiment['name'] in self.parameter['experiments']:
            area_location = self.parameter['area_location']
            length_location = self.parameter['length_location']

            sim[area_location.lower()] = values[0]
            sim[length_location.lower()] = values[1]

        return values, headerValues

    def setupTarget(self):
        area_location = self.parameter['area_location']
        length_location = self.parameter['length_location']
        bound = -1
        comp = -1

        sensitivityOk = 1
        nameArea = area_location.rsplit('/', 1)[-1]
        nameLength = length_location.rsplit('/', 1)[-1]
        unit = int(area_location.split('/')[3].replace('unit_', ''))

        return [(nameArea, unit, comp, bound), (nameLength, unit, comp, bound)], sensitivityOk

    def getBounds(self):
        minVolume = self.parameter['minVolume']
        maxVolume = self.parameter['maxVolume']
        minLength = self.parameter['minLength']
        maxLength = self.parameter['maxLength']

        minValues = [minVolume, minLength]
        maxValues = [maxVolume, maxLength]

        return minValues, maxValues

    def getGradBounds(self):
        return self.getBounds()

    def getHeaders(self):
        bound = -1
        comp = -1
    
        headers = []
        headers.append("Area Comp:%s Bound:%s" % (comp, bound))
        headers.append("Length Comp:%s Bound:%s" % (comp, bound))
        headers.append("Volume Comp:%s Bound:%s" % (comp, bound))
        return headers

    def getHeadersActual(self):
        bound = -1
        comp = -1
    
        headers = []
        headers.append("Volume Comp:%s Bound:%s" % (comp, bound))
        headers.append("Length Comp:%s Bound:%s" % (comp, bound))
        return headers

    def setBounds(self, parameter, lb, ub):
        parameter['minVolume'] = lb[2]
        parameter['maxVolume'] = ub[2]
        parameter['minLength'] = lb[1]
        parameter['maxLength'] = ub[1]

class NormVolumeLengthTransform(VolumeLengthTransform):
    @property
    def name(self):
        return "norm_volume_length"

    def transform(self):
        minVolume = self.parameter['minVolume']
        maxVolume = self.parameter['maxVolume']
        minLength = self.parameter['minLength']
        maxLength = self.parameter['maxLength']

        def trans_volume(i):
            return (i - minVolume)/(maxVolume - minVolume)

        def trans_length(i):
            return (i - minLength)/(maxLength - minLength)

        return [trans_volume, trans_length]

    grad_transform = transform

    def untransform(self, seq):
        minVolume = self.parameter['minVolume']
        maxVolume = self.parameter['maxVolume']
        minLength = self.parameter['minLength']
        maxLength = self.parameter['maxLength']

        minValues = numpy.array([minVolume, minLength])
        maxValues = numpy.array([maxVolume, maxLength])

        values = numpy.array(seq)

        values = (maxValues - minValues) * values + minValues

        values = [values[0]/values[1], values[1]]
        headerValues = [values[0], values[1], values[0]*values[1]]
        return values, headerValues

    def grad_untransform(self, seq):
        return self.untransform(seq)[0]

    def untransform_matrix(self, matrix):
        values = self.untransform_matrix_inputorder(matrix)
        values[:,0] = values[:,0] / values[:,1]
        return values

    def untransform_matrix_inputorder(self, matrix):
        minVolume = self.parameter['minVolume']
        maxVolume = self.parameter['maxVolume']
        minLength = self.parameter['minLength']
        maxLength = self.parameter['maxLength']

        minValues = numpy.array([minVolume, minLength])
        maxValues = numpy.array([maxVolume, maxLength])

        values = (maxValues - minValues) * matrix + minValues

        return values

    def getBounds(self):
        return [0.0, 0.0], [1.0, 1.0]

    def getGradBounds(self):
        return [self.parameter['min'],], [self.parameter['max'],]

plugins = {"norm_volume_length": NormVolumeLengthTransform, "volume_length": VolumeLengthTransform}