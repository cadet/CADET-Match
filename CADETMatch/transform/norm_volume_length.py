import util
import numpy

name = "norm_volume_length"
count = 2
count_extended = 3

def getUnit(location):
    return location.split('/')[3]

def transform(parameter):
    minVolume = parameter['minVolume']
    maxVolume = parameter['maxVolume']
    minLength = parameter['minLength']
    maxLength = parameter['maxLength']

    def trans_volume(i):
        return (i - minVolume)/(maxVolume - minVolume)

    def trans_length(i):
        return (i - minLength)/(maxLength - minLength)

    return [trans_volume, trans_length]

def untransform(seq, cache, parameter, fullPrecision=False):
    minVolume = parameter['minVolume']
    maxVolume = parameter['maxVolume']
    minLength = parameter['minLength']
    maxLength = parameter['maxLength']

    minValues = numpy.array([minVolume, minLength])
    maxValues = numpy.array([maxVolume, maxLength])

    values = numpy.array(seq)

    values = (maxValues - minValues) * values + minValues

    values = [values[0]/values[1], values[1]]

    if cache.roundParameters is not None and not fullPrecision:
        values = [util.RoundToSigFigs(i, cache.roundParameters) for i in values]

    headerValues = [values[0], values[1], values[0]*values[1]]
    return values, headerValues

def setSimulation(sim, parameter, seq, cache, experiment, fullPrecision=False):
    values, headerValues = untransform(seq, cache, parameter, fullPrecision)
    
    if parameter.get('experiments', None) is None or experiment['name'] in parameter['experiments']:
        area_location = parameter['area_location']
        length_location = parameter['length_location']

        sim[area_location.lower()] = values[0]
        sim[length_location.lower()] = values[1]

    return values, headerValues

def setupTarget(parameter):
    area_location = parameter['area_location']
    length_location = parameter['length_location']
    bound = parameter['bound']
    comp = parameter['component']

    sensitivityOk = 1
    nameArea = area_location.rsplit('/', 1)[-1]
    nameLength = length_location.rsplit('/', 1)[-1]
    unit = int(area_location.split('/')[3].replace('unit_', ''))

    return [(nameArea, unit, comp, bound), (nameLength, unit, comp, bound)], sensitivityOk

def getBounds(parameter):
    return [0.0, 0.0], [1.0, 1.0]

def getHeaders(parameter):
    bound = parameter['bound']
    comp = parameter['component']
    
    headers = []
    headers.append("Area Comp:%s Bound:%s" % (comp, bound))
    headers.append("Length Comp:%s Bound:%s" % (comp, bound))
    headers.append("Volume Comp:%s Bound:%s" % (comp, bound))
    return headers

def getHeadersActual(parameter):
    bound = parameter['bound']
    comp = parameter['component']
    
    headers = []
    headers.append("Volume Comp:%s Bound:%s" % (comp, bound))
    headers.append("Length Comp:%s Bound:%s" % (comp, bound))
    return headers

def setBounds(parameter, lb, ub):
    parameter['minVolume'] = lb[2]
    parameter['maxVolume'] = ub[2]
    parameter['minLength'] = lb[1]
    parameter['maxLength'] = ub[1]