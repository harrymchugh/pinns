#!/usr/bin/env python3

import argparse

def parse_args():
    """
    Argument parsing function
    """
    boiler_plate_welcome()

    parser = argparse.ArgumentParser(description="CFD PINN")

    parser.add_argument("--debug",
        action="store_true",
        required=False,
        help="Enable debug output",
        dest="debug")

    #OpenFOAM and geometry args
    parser.add_argument("--case-type",
        action="store",
        type=str,
        required=False,
        choices=["cavity"],
        help="CFDPINN currently only supports OpenFOAM simulations for lid-driven cavity flow",
        dest="case_type")

    parser.add_argument("--case-dir",
        action="store",
        type=str,
        required=False,
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
    
    parser.add_argument("--startx",
        action="store",
        type=float,
        required=False,
        default=0,
        help="The x value of the first cell",
        dest="startx")
    
    parser.add_argument("--starty",
        action="store",
        type=float,
        required=False,
        default=0,
        help="The y value of the first cell",
        dest="starty")
    
    parser.add_argument("--endx",
        action="store",
        type=float,
        required=False,
        default=1,
        help="The x value of the last cell",
        dest="endx")
    
    parser.add_argument("--endy",
        action="store",
        type=float,
        required=False,
        default=1,
        help="The y value of the last cell",
        dest="endy")

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
        required=False,
        help="The viscosity of the fluid used in the OpenFOAM case",
        dest="viscosity")

    parser.add_argument("--initial_u",
        action="store",
        type=float,
        required=False,
        default=1,
        help="The u (x) velocity of the fluid flowing over the cavity top wall",
        dest="initial_u_lid")

    parser.add_argument("--load-simulation",
        action="store_true",
        required=False,
        help="Load simulation data regardless of training or inference",
        dest="load_sim")  

    #Data preprocessing for PINN training
    parser.add_argument("--no-train",
        action="store_true",
        default=False,
        help="Don't train a model, only inference",
        dest="no_train")
            
    parser.add_argument("--test-percent",
        action="store",
        type=float,
        required=False,
        default=0.7,
        help="The percentage of OpenFOAM case data to retain to use as \
            in the testing function of the PINN training process",
        dest="test_size")
    
    parser.add_argument("--save-scaler-path",
        action="store",
        type=str,
        required=False,
        default="",
        help="The output path for the data scalar generated alongside a given input \
            dataset. Required for future use of the model on new data",
        dest="save_scaler_path"
        )
    
    parser.add_argument("--load-scaler-path",
        action="store",
        type=str,
        required=False,
        default="",
        help="The path to load the data scalar for use with inference mode",
        dest="load_scaler_path"
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
    parser.add_argument("--save-model-path",
        action="store",
        type=str,
        required=False,
        default="pinn_model.pt",
        help="Full path for the output of the trained PINN",
        dest="pinn_output_path")
    
    parser.add_argument("--load-model-path",
        action="store",
        type=str,
        help="Path to load a model in inference only mode",
        dest="load_model_path")
    
    parser.add_argument("--profile",
        action="store_true",
        required=False,
        default=False,
        help="Run profiling on train and inference",
        dest="profile")  

    parser.add_argument("--trace-path",
        action="store",
        type=str,
        required=False,
        default="trace.json",
        help="File path for profiling trace json file",
        dest="trace_path")  

    parser.add_argument("--stack-path",
        action="store",
        type=str,
        required=False,
        default="stack.txt",
        help="File path for profiling call-graph file",
        dest="stack_path")   

    parser.add_argument("--tensorboard",
        action="store_true",
        required=False,
        default=False,
        help="Log training metrics with Tensorboard functionality",
        dest="tensorboard")
    
    parser.add_argument("--tensorboard-path",
        action="store",
        type=str,
        required=False,
        default="",
        help="The path to save Tensorboard outputs",
        dest="tensorboard_path") 
    
    parser.add_argument("--epochs",
        action="store",
        type=int,
        required=False,
        default=1,
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
    
    #Need logic for nonsensical combination of args

    return args

def boiler_plate_welcome():
    """
    """
    print("#################")
    print("##  CFD PINN   ##")
    print("#################\n")
