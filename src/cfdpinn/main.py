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
    #if args.training_animation:
        #create_animation(data,args,"train")

    #if args.prediction_animation:
        #create_animation(data,args,"pred")

    #PINN training and testing loop
    pinn.train(data)
    
    #PINN inference
    data = predict_fluid(data,pinn,geom)

    #Save model
    pinn.save_model()

    #Produce plots for analysis
    #if args.static_plots:
        #static_plots(data,args,geom)

if __name__ == "__main__":
    main()
