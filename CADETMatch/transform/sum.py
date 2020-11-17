import CADETMatch.util as util
from CADETMatch.abstract.transform import AbstractTransform


class SumTransform(AbstractTransform):
    @property
    def name(self):
        return "sum"

    @property
    def count(self):
        return 0

    @property
    def count_extended(self):
        return 0

    def transform(self):
        return []

    grad_transform = transform

    def untransform_inputorder(self, seq):
        return []

    def untransform(self, seq):
        return [], []

    def grad_untransform(self, seq):
        return self.untransform(seq)[0]

    def untransform_matrix(self, matrix):
        return None

    untransform_matrix_inputorder = untransform_matrix

    def setSimulation(self, sim, seq, experiment):
        if (
            self.parameter.get("experiments", None) is None
            or experiment["name"] in self.parameter["experiments"]
        ):
            location1 = self.parameter["location1"]
            location2 = self.parameter["location2"]
            locationSum = self.parameter["locationSum"]

            try:
                comp1 = self.parameter["component1"]
                bound1 = self.parameter["bound1"]
                index1 = None
            except KeyError:
                index1 = self.parameter["index1"]
                bound1 = None
                comp1 = None
            value1 = getValue(sim, location1, bound=bound1, comp=comp1, index=index1)

            try:
                comp2 = self.parameter["component2"]
                bound2 = self.parameter["bound2"]
                index2 = None
            except KeyError:
                index2 = self.parameter["index2"]
                bound2 = None
                comp2 = None

            value2 = getValue(sim, location2, bound=bound2, comp=comp2, index=index2)

            try:
                compSum = self.parameter["componentSum"]
                boundSum = self.parameter["boundSum"]
                indexSum = None
            except KeyError:
                indexSum = self.parameter["indexSum"]
                boundSum = None
                compSum = None
            setValue(
                sim,
                value1 + value2,
                locationSum,
                bound=boundSum,
                comp=compSum,
                index=indexSum,
            )

        return [], []

    def getBounds(self):
        return None, None

    def getGradBounds(self):
        return None, None

    def getHeaders(self):
        return []

    def getHeadersActual(self):
        return self.getHeaders()

    def setBounds(self, parameter, lb, ub):
        return None


plugins = {"sum": SumTransform}
