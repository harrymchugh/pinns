#!/usr/bin/env python3

from numpy import zeros
from os import listdir
from Ofpp import parse_internal_field

def load_simulation_data(args, geom):
    """
    
    """
    print("Loading simulation data...")
    data = dict()

    #Get spatio-temporal array shape from geometry
    shape = (
        geom["numt"],
        geom["numy"],
        geom["numx"])
    
    #Create empty arrays to hold input data
    u = zeros(shape)
    v = zeros(shape)
    p = zeros(shape)

    openfoam_outputs = listdir(args.case_dir)
    openfoam_outputs.remove("constant")
    openfoam_outputs.remove("system")
    if "0_orig" in openfoam_outputs:
        openfoam_outputs.remove("0_orig")
    openfoam_outputs.sort()

    spatial_grid_shape = (geom["numy"],geom["numx"])

    idx = 0
    for time in openfoam_outputs:
        openfoam_timestep = args.case_dir + time
        
        if time == "0":
            u = set_initial_condition_u(u,args.initial_u_lid)
            v = set_initial_condition_v(v)
            p = set_initial_condition_p(p)

        else:
            U = parse_internal_field(f"{openfoam_timestep}/U")
            u[idx,:,:] = U[:,0].reshape(spatial_grid_shape)
            v[idx,:,:] = U[:,1].reshape(spatial_grid_shape)
            
            p[idx,:,:] = \
                parse_internal_field(f"{openfoam_timestep}/p").\
                    reshape(spatial_grid_shape)
        
        idx += 1

    data["u"] = u
    data["v"] = v
    data["p"] = p

    print("\tSimulation data loaded\n")

    return data

def set_initial_condition_u(u,velocity):
    """
    Apply the initial conditions for u
    """
    u[0,:,:] = 0
    u[0,-1,:] = velocity
    return u

def set_initial_condition_v(v):
    """
    Apply the initial conditions for v
    """
    v[0,:,:] = 0
    return v

def set_initial_condition_p(p):
    """
    Apply the initial conditions for pressure
    """
    p[0,:,:] = 0
    return p
