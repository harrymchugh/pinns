#!/usr/bin/env python3

import os
from numpy import save

def save_prediction_data(data,args):
    """_summary_

    Args:
        data (_type_): _description_
        args (_type_): _description_
    """
    print("Saving predicted fluid properties...")

    if not os.path.isdir(args.output_data_path):
        os.mkdir(args.output_data_path)
    
    names = [
        "u_pred",
        "v_pred",
        "p_pred",
        "u_residual",
        "v_residual",
        "p_residual"
        ]

    for array_name in names:
        output_file_name = f"{args.output_data_path}/{array_name}"
        save(output_file_name,data[array_name])
    
    print("\tPredicted fluid properties saved\n")

def save_training_data(data,args):
    """_summary_

    Args:
        data (_type_): _description_
        args (_type_): _description_
    """
    print("Saving fluid property training data...")

    if not os.path.isdir(args.output_data_path):
        os.mkdir(args.output_data_path)
    
    names = [
        "u","v","p"
        ]

    for array_name in names:
        output_file_name = f"{args.output_data_path}/{array_name}"
        save(output_file_name,data[array_name])
    
    print("\tFluid property training data saved\n")