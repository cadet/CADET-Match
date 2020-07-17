import CADETMatch.util as util
import numpy
from CADETMatch.abstract.transform import AbstractTransform


class VolumeAreaTransform(AbstractTransform):
    @property
    def name(self):
        return "volume_area"

    @property
    def count(self):
        return 2

    @property
    def count_extended(self):
        return 3

    def transform(self):
        def trans_volume(i):
            return i

        def trans_area(i):
            return i

        return [trans_volume, trans_area]

    grad_transform = transform

    def untransform(self, seq):
        values = [seq[1], seq[0] / seq[1]]
        headerValues = [values[0], values[1], values[0] * values[1]]
        return values, headerValues

    def grad_untransform(self, seq):
        return self.untransform(seq)[0]

    def untransform_matrix(self, matrix):
        temp = self.untransform_matrix_inputorder(matrix)
        values = numpy.zeros(temp.shape)
        values[:, 1] = temp[:, 0] / temp[:, 1]
        values[:, 0] = temp[:, 1]
        return values

    def untransform_matrix_inputorder(self, matrix):
        values = numpy.zeros(matrix.shape)
        values[:, 0] = matrix[:, 0]
        values[:, 1] = matrix[:, 1]
        return values

    def setSimulation(self, sim, seq, experiment):
        values, headerValues = self.untransform(seq)

        if self.parameter.get("experiments", None) is None or experiment["name"] in self.parameter["experiments"]:
            area_location = self.parameter["area_location"]
            length_location = self.parameter["length_location"]

            sim[area_location.lower()] = values[0]
            sim[length_location.lower()] = values[1]

        return values, headerValues

    def getBounds(self):
        minVolume = self.parameter["minVolume"]
        maxVolume = self.parameter["maxVolume"]
        minArea = self.parameter["minArea"]
        maxArea = self.parameter["maxArea"]

        minValues = [minVolume, minArea]
        maxValues = [maxVolume, maxArea]

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
        headers.append("Area")
        return headers

    def setBounds(self, parameter, lb, ub):
        parameter["minVolume"] = lb[2]
        parameter["maxVolume"] = ub[2]
        parameter["minArea"] = lb[0]
        parameter["maxArea"] = ub[0]


class NormVolumeAreaTransform(VolumeAreaTransform):
    @property
    def name(self):
        return "norm_volume_area"

    def transform(self):
        minVolume = self.parameter["minVolume"]
        maxVolume = self.parameter["maxVolume"]
        minArea = self.parameter["minArea"]
        maxArea = self.parameter["maxArea"]

        def trans_volume(i):
            return (i - minVolume) / (maxVolume - minVolume)

        def trans_area(i):
            return (i - minArea) / (maxArea - minArea)

        return [trans_volume, trans_area]

    grad_transform = transform

    def untransform(self, seq):
        minVolume = self.parameter["minVolume"]
        maxVolume = self.parameter["maxVolume"]
        minArea = self.parameter["minArea"]
        maxArea = self.parameter["maxArea"]

        minValues = numpy.array([minVolume, minArea])
        maxValues = numpy.array([maxVolume, maxArea])

        values = numpy.array(seq)

        values = (maxValues - minValues) * values + minValues

        values = [values[1], values[0] / values[1]]
        headerValues = [values[0], values[1], values[0] * values[1]]
        return values, headerValues

    def grad_untransform(self, seq):
        return self.untransform(seq)[0]

    def untransform_matrix(self, matrix):
        temp = self.untransform_matrix_inputorder(matrix)
        values = numpy.zeros(temp.shape)
        values[:, 0] = temp[:, 1]
        values[:, 1] = temp[:, 0] / temp[:, 1]
        return values

    def untransform_matrix_inputorder(self, matrix):
        minVolume = self.parameter["minVolume"]
        maxVolume = self.parameter["maxVolume"]
        minArea = self.parameter["minArea"]
        maxArea = self.parameter["maxArea"]

        minValues = numpy.array([minVolume, minArea])
        maxValues = numpy.array([maxVolume, maxArea])

        temp = (maxValues - minValues) * matrix + minValues

        values = numpy.zeros(temp.shape)
        values[:, 0] = temp[:, 0]
        values[:, 1] = temp[:, 1]

        return values

    def getBounds(self):
        return [0.0, 0.0], [1.0, 1.0]

    def getGradBounds(self):
        return [self.parameter["min"],], [self.parameter["max"],]


plugins = {"norm_volume_area": NormVolumeAreaTransform, "volume_area": VolumeAreaTransform}
