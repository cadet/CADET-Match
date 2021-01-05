#This script generates all the h5 example files
from pathlib import Path
from cadet import H5, Cadet
from addict import Dict
import numpy
import pandas
import CADETMatch.util

import create_sims

def create_experiments(defaults):
    pass

def create_scores(defaults):
    #dextran scores
    dextran_paths = ['DextranShape', 'Shape', 'ShapeBack', 'ShapeFront', 'SSE', 
                     'other/curve', 'other/DextranSSE', 'other/ShapeDecay',
                     'other/ShapeDecayNoDer', 'other/ShapeDecaySimple', 'other/ShapeNoDer', 
                     'other/ShapeOnly', 'other/ShapeSimple', 'other/similarity', 
                     'other/similarityDecay', 'misc/multiple_scores_slicing']

    dex_sim = create_sims.create_dextran_model(defaults)

    scores_dir = defaults.base_dir / "scores"

    for path in dextran_paths:
        dir = scores_dir / path
        dir.mkdir(parents=True, exist_ok=True)
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
    flat_sim.root.input.model.unit_000.sec_000.const_coeff = [1.0,]
    flat_sim.root.input.model.unit_000.sec_001.const_coeff = [1.0,]
    dir = scores_dir / "Ceiling"
    dir.mkdir(parents=True, exist_ok=True)
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
        dir.mkdir(parents=True, exist_ok=True)
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

    dextran_paths = ['gradient', 'graphSpace', 'mcmc/stage1', 'multistart', 'nsga3', 'scoretest',
                     'misc/early_stopping', 'misc/refine_shape', 'misc/refine_sse']

    search_dir = defaults.base_dir / "search"

    for path in dextran_paths:
        dir = search_dir / path
        dir.mkdir(parents=True, exist_ok=True)
        dir_name = dir.name
        dex_sim.filename = (dir / "dextran.h5").as_posix()
        dex_sim.save()
        print("Run ", dir_name, dex_sim.run())
        dex_sim.load()

        times = dex_sim.root.output.solution.solution_times
        values = dex_sim.root.output.solution.unit_002.solution_outlet_comp_000
        data = numpy.array([times, values]).T

        numpy.savetxt(dir / "dextran.csv", data, delimiter=',')


    non_sim = create_sims.create_nonbinding_model(defaults)

    non_paths = ['mcmc/stage2', ]

    search_dir = defaults.base_dir / "search"

    for path in non_paths:
        dir = search_dir / path
        dir.mkdir(parents=True, exist_ok=True)
        dir_name = dir.name
        non_sim.filename = (dir / "non.h5").as_posix()
        non_sim.save()
        print("Run ", dir_name, non_sim.run())
        non_sim.load()

        times = non_sim.root.output.solution.solution_times
        values = non_sim.root.output.solution.unit_002.solution_outlet_comp_000
        data = numpy.array([times, values]).T

        numpy.savetxt(dir / "non.csv", data, delimiter=',')

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
