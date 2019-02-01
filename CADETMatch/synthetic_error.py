import numpy
import copy

def all_steps(simulation):
    "make a switch for each section if it does not have one"
    nsec = simulation.root.input.solver.sections.nsec
    if simulation.root.input.model.connections.nswitches < nsec:
        simulation.root.input.model.connections.nswitches = nsec
        connections = simulation.root.input.model.connections
        for i in range(nsec):
            switch_name = 'switch_%03d' % i
            if switch_name in connections:
                continue
            else:
                switch_name_prev = 'switch_%03d' % (i - 1)
                connections[switch_name] = connections[switch_name_prev].copy()
                connections[switch_name].section = i        

def get_switches(simulation):
    switches = []
    connections = simulation.root.input.model.connections
    nswitches = simulation.root.input.model.connections.nswitches
    
    for i in range(nswitches):
        switch_name = 'switch_%03d' % i
        switches.append(connections[switch_name].copy())
            
    return switches

def iterate_inlets(simulation):
    for key,value in simulation.root.input.model.items():
        if key.startswith('unit_') and value.unit_type == b'INLET':
            yield key, value

def get_inlets(simulation):
    inlets = {}
    nsec = simulation.root.input.solver.sections.nsec
    for unit_name, unit in iterate_inlets(simulation):
        temp = []
        for i in range(nsec):
            sec_name = 'sec_%03d' % i
            temp.append(unit[sec_name].copy())
        inlets[unit_name] = temp
    return inlets
                
def get_section_times(simulation):
    section_times = numpy.diff(simulation.root.input.solver.sections.section_times)
    section_times = [0.0,] + list(section_times)
    return section_times

def update_simulation(simulation, switches, inlets, section_times):
    nsec = len(section_times) - 1
    
    #sections
    simulation.root.input.solver.sections.section_times = section_times
    simulation.root.input.solver.sections.nsec = nsec
    simulation.root.input.solver.sections.section_continuity = [0.0] * (nsec - 1)
    
    #inlets
    for unit_name, unit in inlets.items():
        for idx, sec in enumerate(unit):
            simulation.root.input.model[unit_name]['sec_%03d' % idx] = sec
    
    #switches
    simulation.root.input.model.connections.nswitches = len(switches)
    for idx,item in enumerate(switches):
        simulation.root.input.model.connections['switch_%03d' % idx] = item

def pump_delay(simulation, delays):
    all_steps(simulation)
    
    switches = get_switches(simulation)
    inlets = get_inlets(simulation)
    
    section_times = get_section_times(simulation)
    
    idx = 0
    for delay in delays:
        if delay > 1e-2:
            #add new section time
            section_times.insert(idx+1, delay)
            
            #add new section to all inlets use previous section as a copy
            for inlet in inlets.values():
                cur = copy.deepcopy(inlet[idx])
                
                #set all entries to 0
                for value in cur.values():
                    value[:] = 0
                                
                inlet.insert(idx, cur)
          
            #add a new switch using previous switch connections but all flow 0
            
            #print('switch', idx, len(switches))
            cur = copy.deepcopy(switches[idx])
            
            #set flowrates to 0.0
            cur.connections = []
            
            switches.insert(idx, cur)            
            
            idx += 1
        idx+=1
        
    #fix section numbers
    for idx, i in enumerate(switches):
        i.section = idx
        
    section_times = numpy.cumsum(section_times)
    
    #we can't change the final time point or it will screw up matching, this means that the total time for the simulation needs to be large enough to still work
    section_times[-1] = simulation.root.input.solver.sections.section_times[-1]
    
    #write data to simulation
    
    update_simulation(simulation, switches, inlets, section_times)

def get_flowing_connections(simulation):
    for name,value in simulation.root.input.model.connections.items():
        if name.startswith('switch_') and len(value.connections):
            yield value

def error_flow(simulation, flow):
    for idx,flowing_connection in enumerate(get_flowing_connections(simulation)):
        flowing_connection.connections[4::5] *= flow[idx]

def get_loading_sections(simulation):
    nsec = simulation.root.input.solver.sections.nsec
    for i in range(nsec):
        temp = []
        for key,value in simulation.root.input.model.items():
            if key.startswith('unit_') and value.unit_type == b'INLET':
                temp.append(value['sec_%03d' % i])
        
        if temp:
            if any([numpy.any(i.const_coeff) for i in temp]):
                yield(temp)
      
def error_load(simulation, load):
    for idx,section in enumerate(get_loading_sections(simulation)):
        for inlet in section:
            inlet.const_coeff[:] *= load[idx]
