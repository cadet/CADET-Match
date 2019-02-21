import util

name = "set_value"
count = 0
count_extended = 0

def getUnit(location):
    return location.split('/')[3]

def transform(parameter):
    return []

def untransform(seq, cache, parameter, fullPrecision=False):
    return [], []

def getValue(sim, location, bound=None, comp=None, index=None):
    if bound is not None:
        unit = getUnit(location)
        boundOffset = util.getBoundOffset(sim.root.input.model[unit])

        if comp == -1:
            position = ()
            return sim[location.lower()]
        else:
            position = boundOffset[comp] + bound
            return sim[location.lower()][position]

    if index is not None:
        if index == -1:
            return sim[location.lower()]
        else:
            return sim[location.lower()][index]

def setValue(sim, value, location, bound=None, comp=None, index=None):
    if bound is not None:
        unit = getUnit(location)
        boundOffset = util.getBoundOffset(sim.root.input.model[unit])

        if comp == -1:
            position = ()
            sim[location.lower()] = value
        else:
            position = boundOffset[comp] + bound
            sim[location.lower()][position] = value

    if index is not None:
        if index == -1:
            sim[location.lower()] = value
        else:
            sim[location.lower()][index] = value
        
def setSimulation(sim, parameter, seq, cache, experiment, fullPrecision=False):
    if parameter.get('experiments', None) is None or experiment['name'] in parameter['experiments']:    
        locationFrom = parameter['locationFrom']
        locationTo = parameter['locationTo']
    
        try:
            compFrom = parameter['componentFrom']
            boundFrom = parameter['boundFrom']
            indexFrom = None
        except KeyError:
            indexFrom = parameter['indexFrom']
            boundFrom = None
            compFrom = None
        valueFrom = getValue(sim, locationFrom, bound=boundFrom, comp=compFrom, index=indexFrom)

        try:
            compTo = parameter['componentTo']
            boundTo = parameter['boundTo']
            indexTo = None
        except KeyError:
            indexTo = parameter['indexTo']
            boundTo = None
            compTo = None
        setValue(sim, valueFrom, locationTo, bound=boundTo, comp=compTo, index=indexTo)

    return [],[]

def setupTarget(parameter):
    return [], 0

def getBounds(parameter):
    return None,None

def getHeaders(parameter):
    return []

def getHeadersActual(parameter):
    return []

def setBounds(parameter, lb, ub):
    return None




