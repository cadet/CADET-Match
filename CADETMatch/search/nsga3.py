import CADETMatch.pymoo_config


name = "NSGA3"


def run(cache, tools, creator):
    "run the parameter estimation"
    return CADETMatch.pymoo_config.run(cache, 'nsga3')
