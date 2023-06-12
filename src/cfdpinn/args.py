#!/usr/bin/env python3

import argparse

def parse_args():
    """
    Argument parsing function
    """
    boiler_plate_welcome()

    parser = argparse.ArgumentParser(description="CFD PINN")

    #OpenFOAM and geometry args
    parser.add_argument("--case-type",
        action="store",
        type=str,
        required=True,
        choices=["cavity","channel"],
        help="CFDPINN currently only supports OpenFOAM simulations for lid-driven cavity and channel flow",
        dest="case_type")

    parser.add_argument("--case-dir",
        action="store",
        type=str,
        required=True,
        help="The direcotry of the OpenFOAM case used for PINN training",
        dest="case_dir")

    parser.add_argument("--start-time",
        action="store",
        type=float,
        required=True,
        help="The first value of the OpenFOAM output timesteps",
        dest="start_time")

    parser.add_argument("--end-time",
        action="store",
        type=float,
        required=True,
        help="The last value of the OpenFOAM output timesteps",
        dest="end_time")

    parser.add_argument("--sim-dt",
        action="store",
        type=float,
        required=True,
        help="The original timestep value of the OpenFOAM case",
        dest="sim_dt")
    
    parser.add_argument("--load-dt",
        action="store",
        type=float,
        required=False,
        help="The desired timestep for the loaded OpenFOAM case. \
            This allows the user to decimate the OpenFOAM simulation data. \
            For example a simulation run at 0.01 can be loaded at 0.05 to reduce training time",
        dest="load_dt")
    
    parser.add_argument("--numx",
        action="store",
        type=int,
        required=True,
        help="The number of cells in the x-dimension of the OpenFOAM case",
        dest="numx")

    parser.add_argument("--numy",
        action="store",
        type=int,
        required=True,
        help="The number of cells in the y-dimension of the OpenFOAM case",
        dest="numy")   
    
    parser.add_argument("--viscosity",
        action="store",
        type=float,
        required=True,
        help="The viscosity of the fluid used in the OpenFOAM case",
        dest="viscosity")

    parser.add_argument("--initial_u",
        action="store",
        type=float,
        required=False,
        default=1,
        help="The u (x) velocity of the fluid flowing over the cavity top wall",
        dest="initial_u_lid")

    #Data preprocessing for PINN training
    parser.add_argument("--test-percent",
        action="store",
        type=float,
        required=False,
        default=0.7,
        help="The percentage of OpenFOAM case data to retain to use as \
            in the testing function of the PINN training process",
        dest="test_size")
    
    parser.add_argument("--scaler-path",
        action="store",
        type=str,
        required=True,
        default="",
        help="The output path for the data scalar generated alongside a given input \
            dataset. Required for future use of the model on new data",
        dest="scaler_path"
        )
    
    #MP4 animations
    parser.add_argument("--num-frames",
        action="store",
        type=int,
        required=False,
        default=20,
        help="Number of frames for the fluid animations",
        dest="num_frames")

    parser.add_argument("--training-animation",
        action="store_true",
        required=False,
        default=False,
        help="Output MP4 showing training locations with training \
            U_mag,U,V,P over all timesteps",
        dest="training_animation")
    
    parser.add_argument("--prediction-animation",
        action="store_true",
        required=False,
        default=False,
        help="Output MP4 showing training locations with predicted \
            U_mag,U,V,P over all timesteps",
        dest="prediction_animation")
    
    parser.add_argument("--residual-animation",
        action="store_true",
        required=False,
        default=False,
        help="Output MP4 showing training locations with residual \
            U_mag,U,V,P over all timesteps",
        dest="residual_animation")
    
    #Static plots
    parser.add_argument("--no-static-plots",
        action="store_false",
        required=False,
        default=True,
        help="Stop output of pre-defined static plots for analysis",
        dest="static_plots")
    
    #Output raw training and prediction data
    parser.add_argument("--output-pred-data",
        action="store_true",
        required=False,
        default=False,
        help="Output raw predicted NumPY arrays for U,V,P",
        dest="output_pred_data")

    parser.add_argument("--output-train-data",
        action="store_true",
        required=False,
        default=False,
        help="Output raw training data NumPY arrays for U,V,P",
        dest="output_train_data")

    parser.add_argument("--output-data-path",
        action="store",
        required=False,
        type=str,
        help="Output raw training data NumPY arrays for U,V,P",
        dest="output_data_path")   

    
    #PINN setup and variables
    parser.add_argument("--model-path",
        action="store",
        type=str,
        required=True,
        help="Full path for the output of the trained PINN",
        dest="pinn_output_path")
    
    parser.add_argument("--tensorboard",
        action="store_true",
        required=False,
        default=False,
        help="Log training metrics with Tensorboard functionality",
        dest="tensorboard")
    
    parser.add_argument("--epochs",
        action="store",
        type=int,
        required=True,
        help="The number of epochs to train the PINN",
        dest="epochs")      
    
    parser.add_argument("--lr",
        action="store",
        type=float,
        required=False,
        default=0.001,
        help="The learning rate of the ADAM Optimizer used to train the PINN",
        dest="learning_rate")
    
    #Timing utilities
    parser.add_argument("--inference-timing",
        action="store_true",
        required=False,
        default=False,
        help="Report timings for inference of fluid properties",
        dest="inference_timing")

    #Verbose logging
    parser.add_argument("-v",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose logging",
        dest="verbose")

    args = parser.parse_args()
    
    return args

def boiler_plate_welcome():
    """
    """
    print("#################")
    print("##  CFD PINN   ##")
    print("#################\n")
