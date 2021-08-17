import CADETMatch.pymoo_config


name = "UNSGA3"


def run(cache):
    "run the parameter estimation"
    return CADETMatch.pymoo_config.run(cache, 'unsga3')


