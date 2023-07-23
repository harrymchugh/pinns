#!/usr/bin/env python3

from cfdpinn.args import parse_args
from cfdpinn.geometry import setup_geom
from cfdpinn.inputs import load_simulation_data
from cfdpinn.preprocess import preprocess
from cfdpinn.preprocess import convert_to_tensors
from cfdpinn.preprocess import merge_features
from cfdpinn.pinns import CfdPinn
from cfdpinn.pinns import predict_fluid
from cfdpinn.pinns import compute_residual
from cfdpinn.pinns import load_pinn
from cfdpinn.pinns import save_model
from cfdpinn.plots import static_plots
from cfdpinn.plots import create_animation
from cfdpinn.timing import function_timer
from cfdpinn.outputs import save_prediction_data
from cfdpinn.outputs import save_training_data

from pickle import load

from torch.profiler import profile, record_function, ProfilerActivity
from torch.cuda import is_available
from torch import get_num_threads

def main():
    """
    Driver function of the CFDPINN application
    """
    #Get command line arguments
    args = parse_args()

    #Setup the time-space grids
    geom = setup_geom(args)
    
    #Read in simulation data for training or when 
    #explicitly requested (e.g for generating plots)
    if not args.no_train:
        data = load_simulation_data(args,geom)
    elif args.load_sim:
        data = load_simulation_data(args,geom)

    #PINN setup
    if not args.no_train:
        pinn = CfdPinn(args)
    else: 
        pinn = load_pinn(args)
        
    #Preprocess data for PINN training
    if not args.no_train:
        data = preprocess(data,geom,args)
        data = convert_to_tensors(data,pinn.device)
    
    #Produce animations if necessary
    if args.training_animation:
        create_animation(data,geom,args.num_frames,array_label="train")

    #PINN training and testing loop
    if not args.no_train:
            pinn.train(data,args)
    
    #Inference only mode
    if args.no_train:
        data["scaler"] = load(open(args.load_scaler_path,"rb"))
        data = merge_features(data,geom)
    
    data = predict_fluid(data,pinn,geom,args)
    
    #Create animation from predicted fluid
    if args.prediction_animation:
        create_animation(data,geom,args.num_frames,array_label="pred")

    #Create animation from predicted fluid
    if args.residual_animation:
        data = compute_residual(data)
        create_animation(data,geom,args.num_frames,array_label="residual")

    #Save model and if requested training and predicted fields
    if not args.no_train:
        save_model(pinn)
        
    if args.output_train_data:
        save_training_data(data,args)

    if args.output_pred_data:
        save_prediction_data(data,args)

    #Produce plots for analysis
    if args.static_plots:
        static_plots(data,args,geom)

if __name__ == "__main__":
    main()
