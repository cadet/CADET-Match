import CADETMatch.pymoo_config


name = "UNSGA3"


def run(cache, tools, creator):
    "run the parameter estimation"
    return CADETMatch.pymoo_config.run(cache, 'unsga3')
    

def setupDEAP(
    cache,
    creator,
    base,
    tools,
):
    pass


