import CADETMatch.pymoo_config


name = "NSGA3"


def run(cache):
    "run the parameter estimation"
    return CADETMatch.pymoo_config.run(cache, 'nsga3')
