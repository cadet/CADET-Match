#This script generates all the h5 example files
from pathlib import Path
from cadet import H5, Cadet
from addict import Dict
import numpy
import pandas
import CADETMatch.util

import create_sims

def create_experiments(defaults):
    dex_sim = create_sims.create_dextran_model(defaults)
    non_sim = create_sims.create_nonbinding_model(defaults)

    experiments_dir = defaults.base_dir / "experiments"

    single = experiments_dir / "single"
    single.mkdir(parents=True, exist_ok=True)

    multiple = experiments_dir / "multiple"
    multiple.mkdir(parents=True, exist_ok=True)

    advanced = experiments_dir / "multiple_advanced"
    advanced.mkdir(parents=True, exist_ok=True)

    dex_sim.filename = (single / "dextran.h5").as_posix()
    dex_sim.save()

    dex_sim.filename = (multiple / "dextran1.h5").as_posix()
    dex_sim.save()

    dex_sim.filename = (advanced / "dextran.h5").as_posix()
    dex_sim.save()

    print("Run ", dex_sim.run())
    dex_sim.load()

    times = dex_sim.root.output.solution.solution_times
    values = dex_sim.root.output.solution.unit_002.solution_outlet_comp_000
    data = numpy.array([times, values]).T

    numpy.savetxt(single / "dextran.csv", data, delimiter=',')
    numpy.savetxt(multiple / "dextran1.csv", data, delimiter=',')
    numpy.savetxt(advanced / "dextran.csv", data, delimiter=',')


    dex_sim.filename = (multiple / "dextran2.h5").as_posix()
    dex_sim.root.input.solver.sections.section_times = [0.0, 24.0, 600.0]
    dex_sim.root.input.model.unit_000.sec_000.const_coeff = [0.001,]
    dex_sim.save()

    print("Run ", dex_sim.run())
    dex_sim.load()

    times = dex_sim.root.output.solution.solution_times
    values = dex_sim.root.output.solution.unit_002.solution_outlet_comp_000
    data = numpy.array([times, values]).T

    numpy.savetxt(multiple / "dextran2.csv", data, delimiter=',')

    non_sim.filename = (advanced / "non.h5").as_posix()
    non_sim.save()

    print("Run ", non_sim.run())
    non_sim.load()

    times = non_sim.root.output.solution.solution_times
    values = non_sim.root.output.solution.unit_002.solution_outlet_comp_000
    data = numpy.array([times, values]).T

    numpy.savetxt(advanced / "non.csv", data, delimiter=',')

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

    dirs = [scores_dir / "fractionationSlide", scores_dir / "other" / "fractionationSSE", scores_dir / "misc" / "multiple_components",
           defaults.base_dir / "transforms" / "misc" / "index"]

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
    create_transforms_dextran(defaults)
    create_transforms_non(defaults)
    create_transforms_linear(defaults)
    create_transforms_sum(defaults)
    create_transforms_linear_exp(defaults)

def create_transforms_linear_exp(defaults):
    dex_sim1 = create_sims.create_dextran_model(defaults)

    dex_sim2 = create_sims.create_dextran_model(defaults)
    dex_sim2.root.input.model.unit_001.col_dispersion = 2 * dex_sim2.root.input.model.unit_001.col_dispersion

    dex_sim3 = create_sims.create_dextran_model(defaults)
    dex_sim3.root.input.model.unit_001.col_dispersion = 3 * dex_sim3.root.input.model.unit_001.col_dispersion

    search_dir = defaults.base_dir / "transforms"

    for path in ['norm_linear', 'other/linear']:
        dir = search_dir / path
        dir.mkdir(parents=True, exist_ok=True)
        dir_name = dir.name
        
        dex_sim1.filename = (dir / "dex1.h5").as_posix()
        dex_sim1.save()
        print("Run ", dir_name, dex_sim1.run())
        dex_sim1.load()

        times = dex_sim1.root.output.solution.solution_times
        values = dex_sim1.root.output.solution.unit_002.solution_outlet_comp_000
        data = numpy.array([times, values]).T

        numpy.savetxt(dir / "dex1.csv", data, delimiter=',')

        dex_sim2.filename = (dir / "dex2.h5").as_posix()
        dex_sim2.save()
        print("Run ", dir_name, dex_sim2.run())
        dex_sim2.load()

        times = dex_sim2.root.output.solution.solution_times
        values = dex_sim2.root.output.solution.unit_002.solution_outlet_comp_000
        data = numpy.array([times, values]).T

        numpy.savetxt(dir / "dex2.csv", data, delimiter=',')

        dex_sim3.filename = (dir / "dex3.h5").as_posix()
        dex_sim3.save()
        print("Run ", dir_name, dex_sim3.run())
        dex_sim3.load()

        times = dex_sim3.root.output.solution.solution_times
        values = dex_sim3.root.output.solution.unit_002.solution_outlet_comp_000
        data = numpy.array([times, values]).T

        numpy.savetxt(dir / "dex3.csv", data, delimiter=',')



def create_transforms_sum(defaults):
    cstr_sim = create_sims.create_cstr_model(defaults)

    search_dir = defaults.base_dir / "transforms"

    dir = search_dir / "sum"
    dir.mkdir(parents=True, exist_ok=True)
    dir_name = dir.name
    cstr_sim.filename = (dir / "cstr.h5").as_posix()
    cstr_sim.save()
    print("Run ", dir_name, cstr_sim.run())
    cstr_sim.load()

    times = cstr_sim.root.output.solution.solution_times
    values = cstr_sim.root.output.solution.unit_002.solution_outlet_comp_000
    data = numpy.array([times, values]).T

    numpy.savetxt(dir / "cstr.csv", data, delimiter=',')


def create_transforms_linear(defaults):
    lin_sim = create_sims.create_linear_model(defaults)

    lin_paths = ['auto_keq', 'other/keq', 'other/norm_keq', 'set_value']

    search_dir = defaults.base_dir / "transforms"

    for path in lin_paths:
        dir = search_dir / path
        dir.mkdir(parents=True, exist_ok=True)
        dir_name = dir.name
        lin_sim.filename = (dir / "lin.h5").as_posix()
        lin_sim.save()
        print("Run ", dir_name, lin_sim.run())
        lin_sim.load()

        times = lin_sim.root.output.solution.solution_times
        values = lin_sim.root.output.solution.unit_002.solution_outlet_comp_000
        data = numpy.array([times, values]).T

        numpy.savetxt(dir / "lin.csv", data, delimiter=',')

def create_transforms_non(defaults):
    non_sim = create_sims.create_nonbinding_model(defaults)

    non_paths = ['auto_inverse', 'norm_add', 'norm_mult', ]

    search_dir = defaults.base_dir / "transforms"

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
    
def create_transforms_dextran(defaults):
    dex_sim = create_sims.create_dextran_model(defaults)

    dextran_paths = ['auto', 'norm_diameter', 'norm_volume_area', 'norm_volume_length', 
                     'other/diameter', 'other/log', 'other/norm', 'other/norm_log', 'other/null', 
                     'other/volume_area', 'other/volume_length']

    search_dir = defaults.base_dir / "transforms"

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

def main(defaults):
    "create simulations by directory"
    create_experiments(defaults)
    create_scores(defaults)
    create_search(defaults)
    create_transforms(defaults)

if __name__ == "__main__":
    main()
