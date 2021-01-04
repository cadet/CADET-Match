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

def create_experiments(defaults):
    pass

def create_scores(defaults):
    create_shared_scores(defaults)
    pass

def create_shared_scores(defaults):
    "create all the scores that have the same config except for the score name"
    config = Dict()
    config.CADETPath = Cadet.cadet_path
    config.resultsDir = 'results'
    config.searchMethod = 'NSGA3'
    config.population = defaults.population
    config.gradVector = True
    
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

    dextran_paths = ['DextranShape', 'Shape', 'ShapeBack', 'ShapeFront', 'SSE', 
                     'other/curve', 'other/DextranSSE', 'other/ShapeDecay',
                     'other/ShapeDecayNoDer', 'other/ShapeDecaySimple', 'other/ShapeNoDer', 
                     'other/ShapeOnly', 'other/ShapeSimple', 'other/similarity', 
                     'other/similarityDecay', 'other/width']

    scores_dir = defaults.base_dir / "scores"

    for path in dextran_paths:
        dir = scores_dir / path
        score_name = dir.name
        temp_config = config.deepcopy()
        temp_config.experiments[0].features[0].type = score_name

        create_common(dir, temp_config)


def create_search(defaults):
    config = Dict()
    config.CADETPath = Cadet.cadet_path
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

    #create_nsga3(defaults, config)
    #create_multistart(defaults, config)
    #create_graphspace(defaults, config)
    #create_scoretest(defaults, config)
    #create_gradient(defaults, config)
    #create_early_stopping(defaults, config)
    #create_refine_shape(defaults, config)
    #create_refine_sse(defaults, config)
    #create_altScore(defaults, config)
    #create_mcmc_stage1(defaults, config)
    #create_mcmc_stage2(defaults, config)

def create_common(dir, config):
    config.baseDir =dir.as_posix()

    match_config_file = dir / 'dextran.json'

    with open(match_config_file.as_posix(), 'w') as json_file:
        json.dump(config.to_dict(), json_file, indent='\t')

    #clear the results directory
    results_dir = dir / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)

def create_mcmc_stage1(defaults, config):
    dir = defaults.base_dir / "search" / "mcmc" / "stage1"
    config = config.deepcopy()
    config.searchMethod = 'NSGA3'
    config.population = 12
    config.continueMCMC = 1
    config.MCMCpopulationSet = 12

    error_model = Dict()
    error_model.file_path = "dextran.h5"
    error_model.experimental_csv = "dextran.csv"
    error_model.name = "main"
    error_model.units = [2]
    error_model.delay = [0.0, 2.0]
    error_model.flow = [1.0, 0.001]
    error_model.load = [1.0, 0.001]
    error_model.uv_noise_norm = [1.0, 0.001]

    config.errorModelCount = 1000
    config.errorModel = [error_model,]
    create_common(dir, config)

def create_mcmc_stage2(defaults, config):
    dir = defaults.base_dir / "search" / "mcmc" / "stage2"
    config = config.deepcopy()
    config.baseDir = dir.as_posix()
    config.searchMethod = 'NSGA3'
    config.population = 12
    config.continueMCMC = 1
    config.MCMCpopulationSet = 12

    error_model = Dict()
    error_model.file_path = "non.h5"
    error_model.experimental_csv = "non.csv"
    error_model.name = "main"
    error_model.units = [2]
    error_model.delay = [0.0, 2.0]
    error_model.flow = [1.0, 0.001]
    error_model.load = [1.0, 0.001]
    error_model.uv_noise_norm = [1.0, 0.001]

    config.errorModelCount = 1000
    config.errorModel = [error_model,]

    config.experiments[0].csv = "non.csv"
    config.experiments[0].HDF5 = "non.h5"

    config.experiments[0].features[0].type = "Shape"
    config.experiments[0].features[0].decay = 1

    parameter1 = Dict()
    parameter1.location = '/input/model/unit_001/FILM_DIFFUSION'
    parameter1.min = 1e-12
    parameter1.max = 1e-2
    parameter1.component = -1
    parameter1.bound = -1
    parameter1.transform = 'auto'

    parameter2 = Dict()
    parameter2.location = '/input/model/unit_001/PAR_POROSITY'
    parameter2.min = 0.2
    parameter2.max = 0.7
    parameter2.component = -1
    parameter2.bound = -1
    parameter2.transform = 'auto'

    config.parameters = [parameter1, parameter2]

    mcmc_h5 = Path(defaults.base_dir / "search" / "mcmc" / "stage1" / "results" / "mcmc_refine" / "mcmc" / "mcmc.h5")
    config.mcmc_h5 = mcmc_h5.as_posix()

    match_config_file = dir / 'non.json'

    with open(match_config_file.as_posix(), 'w') as json_file:
        json.dump(config.to_dict(), json_file, indent='\t')

    #clear the results directory
    results_dir = dir / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)

def create_altScore(defaults, config):
    dir = defaults.base_dir / "search" / "misc" / "altScore"
    config = config.deepcopy()
    config.searchMethod = 'AltScore'
    config.population = 0
    config.PreviousResults = (defaults.base_dir / "search" / "nsga3" / "results" / "result.h5").as_posix()
    config.experiments[0].features[0].type = "Shape"
    create_common(dir, config)

def create_refine_shape(defaults, config):
    dir = defaults.base_dir / "search" / "misc" / "refine_shape"
    config = config.deepcopy()
    config.searchMethod = 'NSGA3'
    config.population = defaults.population
    config.gradVector = False
    create_common(dir, config)

def create_refine_sse(defaults, config):
    dir = defaults.base_dir / "search" / "misc" / "refine_sse"
    config = config.deepcopy()
    config.searchMethod = 'NSGA3'
    config.population = defaults.population
    config.gradVector = True
    create_common(dir, config)

def create_early_stopping(defaults, config):
    dir = defaults.base_dir / "search" / "misc" / "early_stopping"
    config = config.deepcopy()
    config.searchMethod = 'NSGA3'
    config.population = defaults.population
    config.gradVector = True
    config.stopAverage = 1e-3
    config.stopBest = 1e-3
    create_common(dir, config)

def create_gradient(defaults, config):
    dir = defaults.base_dir / "search" / "gradient"
    config = config.deepcopy()
    config.searchMethod = 'Gradient'
    config.population = 0
    config.gradVector = 1
    config.experiments[0].features[0].type = "SSE"
    config.seeds = [[2e-7, 0.37],[1e-7, 0.37],[2e-7, 0.41],[1e-6, 0.5],[1e-10, 0.2]]

    create_common(dir, config)

def create_scoretest(defaults, config):
    dir = defaults.base_dir / "search" / "scoretest"
    config = config.deepcopy()
    config.searchMethod = 'ScoreTest'
    config.population = 0
    config.seeds = [[2e-7, 0.37],[1e-7, 0.37],[2e-7, 0.41],[1e-6, 0.5]]

    create_common(dir, config)

def create_graphspace(defaults, config):
    #nsga3
    dir = defaults.base_dir / "search" / "graphSpace"
    config = config.deepcopy()
    config.searchMethod = 'GraphSpace'
    config.population = defaults.population
    config.gradVector = True

    create_common(dir, config)

def create_multistart(defaults, config):
    #nsga3
    dir = defaults.base_dir / "search" / "multistart"
    config = config.deepcopy()
    config.searchMethod = 'Multistart'
    config.population = defaults.population
    config.gradVector = True
    create_common(dir, config)

def create_nsga3(defaults, config):
    #nsga3
    dir = defaults.base_dir / "search" / "nsga3"
    config = config.deepcopy()
    config.searchMethod = 'NSGA3'
    config.population = defaults.population
    config.stallGenerations = 10
    config.finalGradRefinement = True
    config.gradVector = True
    create_common(dir, config)


def create_transforms(defaults):
    pass

def create_typical_experiments(defaults):
    pass

def main(defaults):
    "create simulations by directory"
    create_experiments(defaults)
    create_scores(defaults)
    create_search(defaults)
    create_transforms(defaults)
    create_typical_experiments(defaults)

if __name__ == "__main__":
    main()

