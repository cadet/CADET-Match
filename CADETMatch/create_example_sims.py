from cadet import H5, Cadet
from addict import Dict
import numpy

def set_discretization(model, defaults, n_bound=None):
    columns = {'GENERAL_RATE_MODEL', 'LUMPED_RATE_MODEL_WITH_PORES', 'LUMPED_RATE_MODEL_WITHOUT_PORES'}
    
    
    for unit_name, unit in model.root.input.model.items():
        if 'unit_' in unit_name and unit.unit_type in columns:
            unit.discretization.ncol = defaults.ncol
            unit.discretization.npar = defaults.npar
            
            if n_bound is None:
                n_bound = unit.ncomp*[0]
            unit.discretization.nbound = n_bound
            
            unit.discretization.par_disc_type = 'EQUIDISTANT_PAR'
            unit.discretization.use_analytic_jacobian = 1
            unit.discretization.reconstruction = 'WENO'
            unit.discretization.gs_type = 1
            unit.discretization.max_krylov = 0
            unit.discretization.max_restarts = 10
            unit.discretization.schur_safety = 1.0e-8

            unit.discretization.weno.boundary_model = 0
            unit.discretization.weno.weno_eps = 1e-10
            unit.discretization.weno.weno_order = 3


def get_cadet_template(n_units=3, defaults=None):
    cadet_template = Cadet()
    
    cadet_template.root.input.model.nunits = n_units
    
    # Store solution
    cadet_template.root.input['return'].split_components_data = 1
    cadet_template.root.input['return'].unit_000.write_solution_inlet = 1
    cadet_template.root.input['return'].unit_000.write_solution_outlet = 1
    
    for unit in range(n_units):
        cadet_template.root.input['return']['unit_{0:03d}'.format(unit)] = cadet_template.root.input['return'].unit_000
        
    # Tolerances for the time integrator
    cadet_template.root.input.solver.time_integrator.abstol = defaults.abstol
    cadet_template.root.input.solver.time_integrator.algtol = defaults.algtol
    cadet_template.root.input.solver.time_integrator.reltol = defaults.reltol
    cadet_template.root.input.solver.time_integrator.init_step_size = 1e-6
    cadet_template.root.input.solver.time_integrator.max_steps = 1000000
    
    # Solver settings
    cadet_template.root.input.model.solver.gs_type = 1
    cadet_template.root.input.model.solver.max_krylov = 0
    cadet_template.root.input.model.solver.max_restarts = 10
    cadet_template.root.input.model.solver.schur_safety = 1e-8

    # Run the simulation on single thread
    cadet_template.root.input.solver.nthreads = 0
    
    return cadet_template


def create_dextran_model(defaults):

    dextran_model = get_cadet_template(n_units=3, defaults=defaults)

    # INLET
    dextran_model.root.input.model.unit_000.unit_type = 'INLET'
    dextran_model.root.input.model.unit_000.ncomp = 1
    dextran_model.root.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'

    # Column
    dextran_model.root.input.model.unit_001.unit_type = 'LUMPED_RATE_MODEL_WITH_PORES'
    dextran_model.root.input.model.unit_001.ncomp = 1
    
    dextran_model.root.input.model.unit_001.col_length = 0.25
    dextran_model.root.input.model.unit_001.cross_section_area = 1.0386890710931253E-4
    dextran_model.root.input.model.unit_001.col_porosity = 0.37
    dextran_model.root.input.model.unit_001.par_porosity = 0.33
    dextran_model.root.input.model.unit_001.par_radius = 4.5e-5

    dextran_model.root.input.model.unit_001.col_dispersion = defaults.col_dispersion
    dextran_model.root.input.model.unit_001.film_diffusion = [0.0,]
    
    dextran_model.root.input.model.unit_001.adsorption_model = 'NONE'
    
    dextran_model.root.input.model.unit_001.init_c = [0.0,]
        
    set_discretization(dextran_model, defaults)
    
    ## Outlet
    dextran_model.root.input.model.unit_002.ncomp = 1
    dextran_model.root.input.model.unit_002.unit_type = 'OUTLET'
    
    # Sections and connections
    dextran_model.root.input.solver.sections.nsec = 2
    dextran_model.root.input.solver.sections.section_times = [0.0, 12.0, 600.0]
    dextran_model.root.input.solver.sections.section_continuity = [0,]
    
    ## Inlet Profile
    dextran_model.root.input.model.unit_000.sec_000.const_coeff = [0.002,]
    dextran_model.root.input.model.unit_000.sec_000.lin_coeff = [0.0,]
    dextran_model.root.input.model.unit_000.sec_000.quad_coeff = [0.0,]
    dextran_model.root.input.model.unit_000.sec_000.cube_coeff = [0.0,]


    dextran_model.root.input.model.unit_000.sec_001.const_coeff = [0.0,]
    dextran_model.root.input.model.unit_000.sec_001.lin_coeff = [0.0,]
    dextran_model.root.input.model.unit_000.sec_001.quad_coeff = [0.0,]
    dextran_model.root.input.model.unit_000.sec_001.cube_coeff = [0.0,]
    
    ## Switches
    dextran_model.root.input.model.connections.nswitches = 1
    dextran_model.root.input.model.connections.switch_000.section = 0
    dextran_model.root.input.model.connections.switch_000.connections = [
        0, 1, -1, -1, defaults.flow_rate,
        1, 2, -1, -1, defaults.flow_rate
    ]

    #set the times that the simulator writes out data for
    dextran_model.root.input.solver.user_solution_times = numpy.linspace(0, 600, 601)

    return dextran_model

def create_nonbinding_model(defaults):
    dextran_model = create_dextran_model(defaults)
    dextran_model.root.input.model.unit_001.film_diffusion = [defaults.film_diffusion,]
    dextran_model.root.input.solver.sections.section_times = [0.0, 12.0, 1000.0]
    dextran_model.root.input.solver.user_solution_times = numpy.linspace(0, 1000, 1001)

    return dextran_model

def create_linear_model(defaults):

    linear_model = get_cadet_template(n_units=3, defaults=defaults)

    # INLET
    linear_model.root.input.model.unit_000.unit_type = 'INLET'
    linear_model.root.input.model.unit_000.ncomp = 2
    linear_model.root.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'

    # Column
    linear_model.root.input.model.unit_001.unit_type = 'GENERAL_RATE_MODEL'
    linear_model.root.input.model.unit_001.ncomp = 2
    
    linear_model.root.input.model.unit_001.col_length = 0.25
    linear_model.root.input.model.unit_001.cross_section_area = 1.0386890710931253E-4
    linear_model.root.input.model.unit_001.col_porosity = 0.37
    linear_model.root.input.model.unit_001.par_porosity = 0.33
    linear_model.root.input.model.unit_001.par_radius = 4.5e-5

    linear_model.root.input.model.unit_001.col_dispersion = defaults.col_dispersion
    linear_model.root.input.model.unit_001.film_diffusion = [defaults.film_diffusion,defaults.film_diffusion]
    linear_model.root.input.model.unit_001.par_diffusion = [defaults.par_diffusion, defaults.par_diffusion]
    
    linear_model.root.input.model.unit_001.adsorption_model = 'LINEAR'

    linear_model.root.input.model.unit_001.adsorption.is_kinetic = 1
    linear_model.root.input.model.unit_001.adsorption.lin_ka = [defaults.lin_ka1, defaults.lin_ka2]
    linear_model.root.input.model.unit_001.adsorption.lin_kd = [defaults.lin_kd1, defaults.lin_kd2]
    
    linear_model.root.input.model.unit_001.init_c = [0.0,0.0]
    linear_model.root.input.model.unit_001.init_q = [0.0,0.0]
        
    set_discretization(linear_model, defaults, n_bound=[1,1])
    
    ## Outlet
    linear_model.root.input.model.unit_002.ncomp = 2
    linear_model.root.input.model.unit_002.unit_type = 'OUTLET'
    
    # Sections and connections
    linear_model.root.input.solver.sections.nsec = 2
    linear_model.root.input.solver.sections.section_times = [0.0, 50.0, 1200.0]
    linear_model.root.input.solver.sections.section_continuity = [0,]
    
    ## Inlet Profile
    linear_model.root.input.model.unit_000.sec_000.const_coeff = [0.1,0.1]
    linear_model.root.input.model.unit_000.sec_000.lin_coeff = [0.0,0.0]
    linear_model.root.input.model.unit_000.sec_000.quad_coeff = [0.0,0.0]
    linear_model.root.input.model.unit_000.sec_000.cube_coeff = [0.0,0.0]

    linear_model.root.input.model.unit_000.sec_001.const_coeff = [0.0,0.0]
    linear_model.root.input.model.unit_000.sec_001.lin_coeff = [0.0,0.0]
    linear_model.root.input.model.unit_000.sec_001.quad_coeff = [0.0,0.0]
    linear_model.root.input.model.unit_000.sec_001.cube_coeff = [0.0,0.0]
    
    ## Switches
    linear_model.root.input.model.connections.nswitches = 1
    linear_model.root.input.model.connections.switch_000.section = 0
    linear_model.root.input.model.connections.switch_000.connections = [
        0, 1, -1, -1, defaults.flow_rate,
        1, 2, -1, -1, defaults.flow_rate
    ]

    #set the times that the simulator writes out data for
    linear_model.root.input.solver.user_solution_times = numpy.linspace(0, 1200, 1201)

    return linear_model

def create_cstr_model(defaults):

    cstr_model = get_cadet_template(n_units=5, defaults=defaults)

    # INLET
    cstr_model.root.input.model.unit_000.unit_type = 'INLET'
    cstr_model.root.input.model.unit_000.ncomp = 1
    cstr_model.root.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'

    # CSTR
    cstr_model.root.input.model.unit_001.unit_type = 'CSTR'
    cstr_model.root.input.model.unit_001.ncomp = 1
    cstr_model.root.input.model.unit_001.init_c = [0.0]
    cstr_model.root.input.model.unit_001.adsorption_model = 'NONE'
    cstr_model.root.input.model.unit_001.init_volume = 5e-6

    cstr_model.root.input.model.unit_003.unit_type = 'CSTR'
    cstr_model.root.input.model.unit_003.ncomp = 1
    cstr_model.root.input.model.unit_003.init_c = [0.0]
    cstr_model.root.input.model.unit_003.adsorption_model = 'NONE'
    cstr_model.root.input.model.unit_003.init_volume = 5e-6

    cstr_model.root.input.model.unit_004.unit_type = 'CSTR'
    cstr_model.root.input.model.unit_004.ncomp = 1
    cstr_model.root.input.model.unit_004.init_c = [0.0]
    cstr_model.root.input.model.unit_004.adsorption_model = 'NONE'
    cstr_model.root.input.model.unit_004.init_volume = 1e-5

    
    ## Outlet
    cstr_model.root.input.model.unit_002.ncomp = 1
    cstr_model.root.input.model.unit_002.unit_type = 'OUTLET'
    
    # Sections and connections
    cstr_model.root.input.solver.sections.nsec = 2
    cstr_model.root.input.solver.sections.section_times = [0.0, 50.0, 1200.0]
    cstr_model.root.input.solver.sections.section_continuity = [0,]
    
    ## Inlet Profile
    cstr_model.root.input.model.unit_000.sec_000.const_coeff = [0.1]
    cstr_model.root.input.model.unit_000.sec_000.lin_coeff = [0.0,]
    cstr_model.root.input.model.unit_000.sec_000.quad_coeff = [0.0,]
    cstr_model.root.input.model.unit_000.sec_000.cube_coeff = [0.0,]

    cstr_model.root.input.model.unit_000.sec_001.const_coeff = [0.0]
    cstr_model.root.input.model.unit_000.sec_001.lin_coeff = [0.0,]
    cstr_model.root.input.model.unit_000.sec_001.quad_coeff = [0.0,]
    cstr_model.root.input.model.unit_000.sec_001.cube_coeff = [0.0,]
    
    ## Switches
    cstr_model.root.input.model.connections.nswitches = 1
    cstr_model.root.input.model.connections.switch_000.section = 0
    cstr_model.root.input.model.connections.switch_000.connections = [
        0, 1, -1, -1, defaults.flow_rate,
        1, 3, -1, -1, defaults.flow_rate,
        3, 2, -1, -1, defaults.flow_rate,
        0, 4, -1, -1, defaults.flow_rate,
        4, 2, -1, -1, defaults.flow_rate
    ]

    #set the times that the simulator writes out data for
    cstr_model.root.input.solver.user_solution_times = numpy.linspace(0, 1200, 1201*10)

    return cstr_model
