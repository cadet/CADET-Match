#This script generates all the h5 example files
from pathlib import Path
from cadet import H5, Cadet
from addict import Dict
import numpy
import pandas
import json
import shutil
import CADETMatch.util

import create_sims

defaults = Dict()
defaults.cadet_path = Path(r"C:\Users\kosh_000\cadet_build\CADET\CADET41\bin\cadet-cli.exe").as_posix()
defaults.base_dir = Path(__file__).parent
defaults.population = 20

Cadet.cadet_path = defaults.cadet_path


def create_experiments(defaults):
    pass

def create_scores(defaults):
    pass

def create_search(defaults):
    config = Dict()
    config.CADETPath = Cadet.cadet_path
    #config.baseDir = base_dir.as_posix()
    config.resultsDir = 'results'
    
    parameter1 = Dict()
    parameter1.location = '/input/model/unit_001/COL_DISPERSION'
    parameter1.min = 1e-10
    parameter1.max = 1e-6
    parameter1.component = -1
    parameter1.bound = -1
    parameter1.transform = 'auto'

    parameter2 = Dict()
    parameter2.location = '/input/model/unit_001/COL_POROSITY'
    parameter2.min = 0.2
    parameter2.max = 0.7
    parameter2.component = -1
    parameter2.bound = -1
    parameter2.transform = 'auto'

    config.parameters = [parameter1, parameter2]

    experiment1 = Dict()
    experiment1.name = 'main'
    experiment1.csv = 'dextran.csv'
    experiment1.HDF5 = 'dextran.h5'
    experiment1.isotherm = '/output/solution/unit_002/SOLUTION_OUTLET_COMP_000'

    config.experiments = [experiment1,]

    feature1 = Dict()
    feature1.name = "main_feature"
    feature1.type = 'DextranShape'

    experiment1.features = [feature1,]

    create_nsga3(config)
    create_multistart(config)
    create_graphspace(config)
    create_scoretest(config)
    create_gradient(config)
    create_early_stopping(config)
    create_refine_shape(config)
    create_refine_sse(config)
    create_altScore(config)

def create_search_common(dir, config):
    config.baseDir =dir.as_posix()

    match_config_file = dir / 'dextran.json'

    with open(match_config_file.as_posix(), 'w') as json_file:
        json.dump(config.to_dict(), json_file, indent='\t')

    #clear the results directory
    results_dir = dir / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)

def create_altScore(config):
    dir = defaults.base_dir / "search" / "misc" / "altScore"
    config = config.deepcopy()
    config.searchMethod = 'AltScore'
    config.population = 0
    config.PreviousResults = (defaults.base_dir / "search" / "nsga3" / "results" / "result.h5").as_posix()
    config.experiments[0].features[0].type = "Shape"
    create_search_common(dir, config)

def create_refine_shape(config):
    dir = defaults.base_dir / "search" / "misc" / "refine_shape"
    config = config.deepcopy()
    config.searchMethod = 'NSGA3'
    config.population = defaults.population
    config.gradVector = False
    create_search_common(dir, config)

def create_refine_sse(config):
    dir = defaults.base_dir / "search" / "misc" / "refine_sse"
    config = config.deepcopy()
    config.searchMethod = 'NSGA3'
    config.population = defaults.population
    config.gradVector = True
    create_search_common(dir, config)

def create_early_stopping(config):
    dir = defaults.base_dir / "search" / "misc" / "early_stopping"
    config = config.deepcopy()
    config.searchMethod = 'NSGA3'
    config.population = defaults.population
    config.gradVector = True
    config.stopAverage = 1e-3
    config.stopBest = 1e-3
    create_search_common(dir, config)

def create_gradient(config):
    dir = defaults.base_dir / "search" / "gradient"
    config = config.deepcopy()
    config.searchMethod = 'Gradient'
    config.population = 0
    config.gradVector = 1
    config.experiments[0].features[0].type = "SSE"
    config.seeds = [[2e-7, 0.37],[1e-7, 0.37],[2e-7, 0.41],[1e-6, 0.5],[1e-10, 0.2]]

    create_search_common(dir, config)

def create_scoretest(config):
    dir = defaults.base_dir / "search" / "scoretest"
    config = config.deepcopy()
    config.searchMethod = 'ScoreTest'
    config.population = 0
    config.seeds = [[2e-7, 0.37],[1e-7, 0.37],[2e-7, 0.41],[1e-6, 0.5]]

    create_search_common(dir, config)

def create_graphspace(config):
    #nsga3
    dir = defaults.base_dir / "search" / "graphSpace"
    config = config.deepcopy()
    config.searchMethod = 'GraphSpace'
    config.population = defaults.population
    config.gradVector = True

    create_search_common(dir, config)

def create_multistart(config):
    #nsga3
    dir = defaults.base_dir / "search" / "multistart"
    config = config.deepcopy()
    config.searchMethod = 'Multistart'
    config.population = defaults.population
    config.gradVector = True
    create_search_common(dir, config)

def create_nsga3(config):
    #nsga3
    dir = defaults.base_dir / "search" / "nsga3"
    config = config.deepcopy()
    config.searchMethod = 'NSGA3'
    config.population = defaults.population
    config.stallGenerations = 10
    config.finalGradRefinement = True
    config.gradVector = True
    create_search_common(dir, config)


def create_transforms(defaults):
    pass

def create_typical_experiments(defaults):
    pass

def main():
    "create simulations by directory"
    create_experiments(defaults)
    create_scores(defaults)
    create_search(defaults)
    create_transforms(defaults)
    create_typical_experiments(defaults)

if __name__ == "__main__":
    main()

