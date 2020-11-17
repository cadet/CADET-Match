import numpy

import CADETMatch.util as util
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

    def untransform_inputorder(self, seq):
        volume = seq[0]
        length = seq[1]
        return [volume, length]

    def untransform(self, seq):
        volume, length = self.untransform_inputorder(seq)
        area = volume / length
        values = [area, length]
        headerValues = [area, length, volume]
        return values, headerValues

    def grad_untransform(self, seq):
        return self.untransform(seq)[0]

    def untransform_matrix(self, matrix):
        values = self.untransform_matrix_inputorder(matrix)
        values[:, 0] = values[:, 0] / values[:, 1]
        return values

    def untransform_matrix_inputorder(self, matrix):
        values = numpy.zeros(matrix.shape)
        values[:, 0] = matrix[:, 0]
        values[:, 1] = matrix[:, 1]
        return values

    def setSimulation(self, sim, seq, experiment):
        values, headerValues = self.untransform(seq)

        if (
            self.parameter.get("experiments", None) is None
            or experiment["name"] in self.parameter["experiments"]
        ):
            area_location = self.parameter["area_location"]
            length_location = self.parameter["length_location"]

            sim[area_location.lower()] = values[0]
            sim[length_location.lower()] = values[1]

        return values, headerValues

    def getBounds(self):
        minVolume = self.parameter["minVolume"]
        maxVolume = self.parameter["maxVolume"]
        minLength = self.parameter["minLength"]
        maxLength = self.parameter["maxLength"]

        minValues = [minVolume, minLength]
        maxValues = [maxVolume, maxLength]

        return minValues, maxValues

    def getGradBounds(self):
        return self.getBounds()

    def getHeaders(self):
        headers = []
        headers.append("Area")
        headers.append("Length")
        headers.append("Volume")
        return headers

    def getHeadersActual(self):
        headers = []
        headers.append("Volume")
        headers.append("Length")
        return headers

    def setBounds(self, parameter, lb, ub):
        parameter["minVolume"] = lb[0]
        parameter["maxVolume"] = ub[0]
        parameter["minLength"] = lb[1]
        parameter["maxLength"] = ub[1]


class NormVolumeLengthTransform(VolumeLengthTransform):
    @property
    def name(self):
        return "norm_volume_length"

    def transform(self):
        minVolume = self.parameter["minVolume"]
        maxVolume = self.parameter["maxVolume"]
        minLength = self.parameter["minLength"]
        maxLength = self.parameter["maxLength"]

        def trans_volume(i):
            return (i - minVolume) / (maxVolume - minVolume)

        def trans_length(i):
            return (i - minLength) / (maxLength - minLength)

        return [trans_volume, trans_length]

    grad_transform = transform

    def untransform_inputorder(self, seq):
        minVolume = self.parameter["minVolume"]
        maxVolume = self.parameter["maxVolume"]
        minLength = self.parameter["minLength"]
        maxLength = self.parameter["maxLength"]

        minValues = numpy.array([minVolume, minLength])
        maxValues = numpy.array([maxVolume, maxLength])

        values = numpy.array(seq)

        values = (maxValues - minValues) * values + minValues
        return values

    def grad_untransform(self, seq):
        return self.untransform(seq)[0]

    def untransform_matrix(self, matrix):
        values = self.untransform_matrix_inputorder(matrix)
        values[:, 0] = values[:, 0] / values[:, 1]
        return values

    def untransform_matrix_inputorder(self, matrix):
        minVolume = self.parameter["minVolume"]
        maxVolume = self.parameter["maxVolume"]
        minLength = self.parameter["minLength"]
        maxLength = self.parameter["maxLength"]

        minValues = numpy.array([minVolume, minLength])
        maxValues = numpy.array([maxVolume, maxLength])

        values = (maxValues - minValues) * matrix + minValues

        return values

    def getBounds(self):
        return [0.0, 0.0], [1.0, 1.0]

    def getGradBounds(self):
        return [self.parameter["min"],], [
            self.parameter["max"],
        ]


plugins = {
    "norm_volume_length": NormVolumeLengthTransform,
    "volume_length": VolumeLengthTransform,
}
