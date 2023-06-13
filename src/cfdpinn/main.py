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
    
    #Read in simulation data
    if not args.no_train:
        data = load_simulation_data(args,geom)

    #PINN setup
    if not args.no_train:
        pinn = CfdPinn(args)
    else: 
        pinn = load_pinn(args)
        pinn.device = 'cuda' if is_available() else 'cpu'
        pinn.to(pinn.device)
        if args.debug:
            print(f"DEBUG: PINN device: {pinn.device}\n")
            print(f"DEBUG: threads: {get_num_threads()}")

    #Preprocess data for PINN training
    if not args.no_train:
        data = preprocess(data,geom,args)
        data = convert_to_tensors(data,pinn.device)
    
    #Produce animations if necessary
    if args.training_animation:
        create_animation(data,geom,args.num_frames,array_label="train")

    #PINN training and testing loop
    if not args.no_train:
        if args.profile:
            with profile(
                activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
                profile_memory=True,
                with_stack=True,
                record_shapes=True) as prof:
                with record_function("model_training"):
                    pinn.train(data)
            
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
            print()
            print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
            print()
            print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cpu_time_total", row_limit=10))
            print()

            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            print()
            print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
            print()
            print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=10))
            print()


            prof.export_chrome_trace(args.trace_path)
            prof.export_stacks(f"cpu_{args.stack_path}", "self_cpu_time_total")
            prof.export_stacks(f"gpu_{args.stack_path}", "self_cuda_time_total")
        
        else:
            pinn.train(data)
    
    #Inference only mode
    if args.no_train:
        data = dict()
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
