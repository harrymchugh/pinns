#!/usr/bin/env python3

from numpy import arange
from numpy import meshgrid
from numpy import linspace
from math import isclose
from decimal import Decimal

def setup_geom(args):
    """
    Return a dictionary holding all geometry information
    for the input simulation data
    """
    print("Setting up geometry...")

    geom = dict()
    geom["x_start"] = 0
    geom["x_end"] = 1

    if args.numx < 0:
        msg = "The --numx arg value must be positive"
        Exception(msg)
    geom["numx"] = args.numx
    
    geom["y_start"] = 0
    geom["y_end"] = 1

    if args.numy < 0:
        msg = "The --numy arg value must be positive"
        Exception(msg)
    geom["numy"] = args.numy

    if args.start_time < 0:
        msg = "The --start-time arg value must be positive"
        Exception(msg)

    if args.start_time > args.end_time:
        msg = "The --start-time arg value must be less than --end-time"
        Exception(msg)

    geom["t_start"] = args.start_time
    geom["t_end"] = args.end_time
    
    if args.sim_dt < 0:
        msg = "The --sim-dt arg cannot be negative"
        Exception(msg)
    
    geom["t_dt"] = args.sim_dt
    
    if args.load_dt:
        #Check the load_dt is suitable
        if args.load_dt < args.sim_dt:
            msg = "The --load-dt arg value must be greater than --sim-dt"
            Exception(msg)

        if args.load_dt < 0:
            msg = "The --load-dt arg cannot be negative"
            Exception(msg)
        
        mod = Decimal(args.load_dt) % Decimal(args.sim_dt)
        if isclose(mod,0,rel_tol=1e-6):
            msg = "--load-dt is not divisible by --sim-dt. Decimation is not possible"
            Exception(msg)
        else:
            geom["stride"] = Decimal(args.load_dt) / Decimal(args.sim_dt)
            geom["t_dt"] = args.load_dt

        geom["numt"] = len(
            arange(
                geom["t_start"],
                geom["t_end"] + geom["t_dt"],
                geom["t_dt"])
            )
    
    geom["grid2d_x"],geom["grid2d_y"] = meshgrid(
        linspace(geom["x_start"],geom["x_end"],geom["numx"]),
        linspace(geom["y_start"],geom["y_end"],geom["numy"]))
    
    print("\tGeometry setup completed\n")

    return geom