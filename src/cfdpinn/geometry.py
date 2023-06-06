#!/usr/bin/env python3

from numpy import arange
from numpy import meshgrid
from numpy import linspace

def setup_geom(args):
    """
    Return a dictionary holding all geometry information
    for the input simulation data
    """
    geom = dict()
    geom["x_start"] = 0
    geom["x_end"] = 1
    geom["numx"] = args.numx
    
    geom["y_start"] = 0
    geom["y_end"] = 1
    geom["numy"] = args.numy

    geom["t_start"] = args.start_time
    geom["t_end"] = args.end_time
    geom["t_dt"] = args.dt
    geom["numt"] = len(
        arange(
            geom["t_start"],
            geom["t_end"] + geom["t_dt"],
            geom["t_dt"])
        )
    
    geom["grid2d_x"],geom["grid2d_y"] = meshgrid(
        linspace(geom["x_start"],geom["x_end"],geom["numx"]),
        linspace(geom["y_start"],geom["y_end"],geom["numy"]))
    
    return geom