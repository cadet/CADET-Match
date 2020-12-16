#This script generates all the h5 example files
from pathlib import Path
from cadet import H5, Cadet
from addict import Dict
import numpy
import pandas
import CADETMatch.util

import create_sims

defaults = Dict()
defaults.cadet_path = Path(r"C:\Users\kosh_000\cadet_build\CADET\CADET41\bin\cadet-cli.exe").as_posix()
defaults.base_dir = Path(__file__).parent
defaults.flow_rate = 2.88e-8 # m^3/s
defaults.ncol = 100
defaults.npar = 10
defaults.abstol = 1e-8
defaults.algtol = 1e-10
defaults.reltol = 1e-8
defaults.lin_ka1 = 4e-4
defaults.lin_ka2 = 1e-4
defaults.lin_kd1 = 4e-3
defaults.lin_kd2 = 1e-3
defaults.col_dispersion = 2e-7
defaults.film_diffusion = 1e-6
defaults.par_diffusion = 3e-11

Cadet.cadet_path = defaults.cadet_path


def create_experiments(defaults):
    pass

def create_scores(defaults):
    #dextran scores
    dextran_paths = ['DextranShape', 'Shape', 'ShapeBack', 'ShapeFront', 'SSE', 
                     'other/curve', 'other/DextranSSE', 'other/LogSSE', 'other/ShapeDecay',
                     'other/ShapeDecayNoDer', 'other/ShapeDecaySimple', 'other/ShapeNoDer', 
                     'other/ShapeOnly', 'other/ShapeSimple', 'other/similarity', 
                     'other/similarityDecay', 'other/width', 'misc/slicing', 
                     'misc/multiple_scores']

    dex_sim = create_sims.create_dextran_model(defaults)

    scores_dir = defaults.base_dir / "scores"

    for path in dextran_paths:
        dir = scores_dir / path
        dir_name = dir.name
        dex_sim.filename = (dir / "dextran.h5").as_posix()
        dex_sim.save()
        print("Run ", dir_name, dex_sim.run())
        dex_sim.load()

        times = dex_sim.root.output.solution.solution_times
        values = dex_sim.root.output.solution.unit_002.solution_outlet_comp_000
        data = numpy.array([times, values]).T

        numpy.savetxt(dir / "dextran.csv", data, delimiter=',')

    #flatten
    flat_sim = create_sims.create_dextran_model(defaults)
    flat_sim.root.input.model.unit_000.sec_000.const_coeff = [0.0,]
    dir = scores_dir / "Ceiling"
    dir_name = dir.name
    flat_sim.filename = (dir / "flat.h5").as_posix()
    flat_sim.save()
    print("Run ", dir_name, flat_sim.run())
    flat_sim.load()

    times = flat_sim.root.output.solution.solution_times
    values = flat_sim.root.output.solution.unit_002.solution_outlet_comp_000
    data = numpy.array([times, values]).T

    numpy.savetxt(dir / "flat.csv", data, delimiter=',')


    #fractionate
    #use a 2-comp linear isotherm and densely sample the time to improve the estimation
    frac_sim = create_sims.create_linear_model(defaults)

    dirs = [scores_dir / "fractionationSlide", scores_dir / "other" / "fractionationSSE", scores_dir / "misc" / "multiple_components"]

    for dir in dirs:
        dir_name = dir.name
        frac_sim.filename = (dir / "fraction.h5").as_posix()
        frac_sim.save()
        print("Run ", dir_name, frac_sim.run())
        frac_sim.load()

        times0 = frac_sim.root.output.solution.solution_times
        values0 = frac_sim.root.output.solution.unit_002.solution_outlet_comp_000
        data0 = numpy.array([times0, values0]).T

        values1 = frac_sim.root.output.solution.unit_002.solution_outlet_comp_001
        data1 = numpy.array([times0, values1]).T

        data_sum = numpy.array([times0, values0 + values1]).T

        numpy.savetxt(dir / "comp0.csv", data0, delimiter=',')
        numpy.savetxt(dir / "comp1.csv", data1, delimiter=',')
        numpy.savetxt(dir / "data_sum.csv", data_sum, delimiter=',')

        times = numpy.linspace(400, 800, 9)
        start_times = times[:-1]
        stop_times = times[1:]
        fracs = CADETMatch.util.fractionate_sim(start_times, stop_times, [0,1], frac_sim, 'unit_002')

        df = pandas.DataFrame({'Start':start_times, 'Stop':stop_times, '0':fracs[0], '1':fracs[1]})
        df.to_csv((dir / "frac.csv").as_posix(), index=False, header=True)


def create_search(defaults):
    dex_sim = create_sims.create_dextran_model(defaults)

    dextran_paths = ['gradient', 'graphSpace', 'mcmc', 'multistart', 'nsga3', 'scoretest',
                     'misc/early_stopping', 'misc/refine_shape', 'misc/refine_sse']

    search_dir = defaults.base_dir / "search"

    for path in dextran_paths:
        dir = search_dir / path
        dir_name = dir.name
        dex_sim.filename = (dir / "dextran.h5").as_posix()
        dex_sim.save()
        print("Run ", dir_name, dex_sim.run())
        dex_sim.load()

        times = dex_sim.root.output.solution.solution_times
        values = dex_sim.root.output.solution.unit_002.solution_outlet_comp_000
        data = numpy.array([times, values]).T

        numpy.savetxt(dir / "dextran.csv", data, delimiter=',')

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
