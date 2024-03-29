#!/usr/bin/env python3

import argparse

def parse_args():
    """
    A function to capture command line arguments
    to control program logic
    """
    boiler_plate_welcome()

    parser = argparse.ArgumentParser(description="CFD PINN")

    parser.add_argument("--debug",
        action="store_true",
        required=False,
        help="Enable debug output",
        dest="debug")

    ## OpenFOAM case setup
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
        help="The directory of the OpenFOAM case used for PINN training",
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
    
    ##Animation and plots
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

    parser.add_argument("--animations-path",
        action="store",
        type=str,
        required=False,
        default="",
        help="The output path for the animations",
        dest="animations_path"
        )

    parser.add_argument("--static-plots",
        action="store_true",
        required=False,
        default=False,
        help="Produce static plots for analysis",
        dest="static_plots")

    parser.add_argument("--static-plots-path",
        action="store",
        type=str,
        required=False,
        default="",
        help="The output path for the static plots",
        dest="static_plots_path"
        )

    ## PINNs
    parser.add_argument("--load-simulation",
        action="store_true",
        required=False,
        help="Load simulation data regardless of training or inference",
        dest="load_sim")  

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
    
    parser.add_argument("--device",
        action="store",
        type=str,
        choices=["cuda","cpu"],
        required=False,
        default="cuda",
        help="The type of device used for PyTorch operations, default is to use GPU is available \
            but fall back to CPU if a GPU cannot be found",
        dest="device")
    
    parser.add_argument("--save-model-path",
        action="store",
        type=str,
        required=False,
        default="pinn_model.pt",
        help="Full path for the output of the trained PINN",
        dest="pinn_output_path")
    
    parser.add_argument("--optimizer",
        action="store",
        type=str,
        choices=["adam","sgd"],
        required=False,
        default="adam",
        help="The type of optimizer used for model training",
        dest="optimizer")
    
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

    parser.add_argument("--profile-path",
        action="store",
        type=str,
        required=False,
        default="",
        help="The path to save Tensorboard profiling outputs",
        dest="profile_path") 

    parser.add_argument("--tensorboard",
        action="store_true",
        required=False,
        default=False,
        help="Log training adn testing metrics with Tensorboard functionality",
        dest="tensorboard")
    
    parser.add_argument("--tensorboard-path",
        action="store",
        type=str,
        required=False,
        default="",
        help="The path to save Tensorboard training/testing metrics",
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
    
    parser.add_argument("--inference-timing",
        action="store_true",
        required=False,
        default=False,
        help="Report timings for inference of fluid properties",
        dest="inference_timing")

    parser.add_argument("--adaption",
        action="store",
        type=str,
        choices=["lrannealing","softadapt","noadaption"],
        required=False,
        default="noadaption",
        help="The type of loss function weighting to use",
        dest="adaption")

    args = parser.parse_args()
    
    if args.debug:
        print(args)

    return args

def boiler_plate_welcome():
    """
    A boiler plate welcome to the program
    """
    print("#################")
    print("##  CFD PINN   ##")
    print("#################\n")
