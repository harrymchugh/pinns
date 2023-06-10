#!/usr/bin/env python3

from cfdpinn.args import parse_args
from cfdpinn.geometry import setup_geom
from cfdpinn.inputs import load_simulation_data
from cfdpinn.preprocess import preprocess
from cfdpinn.preprocess import convert_to_tensors
from cfdpinn.pinns import CfdPinn
from cfdpinn.pinns import predict_fluid
from cfdpinn.plots import static_plots
from cfdpinn.plots import create_animation
from cfdpinn.timing import function_timer
from cfdpinn.outputs import save_prediction_data
from cfdpinn.outputs import save_training_data

def main():
    """
    Driver function of the CFDPINN application
    """
    #Get command line arguments
    args = parse_args()

    #Setup the time-space grids
    geom = setup_geom(args)
    
    #Read in simulation data
    data = load_simulation_data(args,geom)

    #PINN setup
    pinn = CfdPinn(args)

    #Preprocess data for PINN training
    data = preprocess(data,geom,args)
    data = convert_to_tensors(data,pinn.device)
    
    #Produce animations if necessary
    if args.training_animation:
        create_animation(data,geom,args.num_frames,array_label="train")

    #PINN training and testing loop
    pinn.train(data)
    
    #PINN inference
    data = predict_fluid(data,pinn,geom,args)

    #Create animation from predicted fluid
    if args.prediction_animation:
        create_animation(data,geom,args.num_frames,array_label="pred")

    #Create animation from predicted fluid
    if args.residual_animation:
        create_animation(data,geom,args.num_frames,array_label="residual")

    #Save model and if requested training and predicted fields
    pinn.save_model()
    if args.output_train_data:
        save_training_data(data,args)

    if args.output_pred_data:
        save_prediction_data(data,args)

    #Produce plots for analysis
    if args.static_plots:
        static_plots(data,args,geom)

if __name__ == "__main__":
    main()
