#!/usr/bin/env python3

import torch

from numpy import absolute
from numpy import zeros
from numpy import min,max,median,std

from tqdm import tqdm

from cfdpinn.timing import function_timer

from time import time

from softadapt import LossWeightedSoftAdapt

from torch.utils.tensorboard import SummaryWriter

class CfdPinn(torch.nn.Module):
    def __init__(self,args):
        """
        The class for the CFDPINN neural networks.
        """
        super().__init__()
        self.linear_stack = torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3)
        )

        if args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)
        elif args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=args.learning_rate)
        else:
            msg = f"Somehow you have managed to pass an --optimizer name that is not supported"
            raise Exception(msg)

        self.criterion = torch.nn.MSELoss()
        self.viscosity = args.viscosity
        self.tensorboard_path = args.tensorboard_path

        if args.tensorboard:
            self.writer = SummaryWriter(self.tensorboard_path)

        self.adaption = args.adaption
        self.epochs = args.epochs
        self.alpha = 0.9

        self.model_output_path = args.pinn_output_path
        self.choose_model_device(args)

    def choose_model_device(self,args):
        """
        A function to set PyTorch training and inference device.

        If a GPU is available it will be used, unless the user
        has specifically asked for the CPU with --device cpu at 
        the command line
        """
        if args.device == "cpu":
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.to(self.device)
        
        if args.debug:
            print(f"DEBUG: PINN device: {self.device}\n")

    def lossfn(self,data,train_or_test):
        """
        A loss function that considers 2D Navier-Stokes and 
        returns multi-objective components and applies
        weighting if necessary.
        """
        #Get gradients required for 2D-Navier stokes
        u = data[f"{train_or_test}_interior_pred"][:,0]
        v = data[f"{train_or_test}_interior_pred"][:,1]
        p = data[f"{train_or_test}_interior_pred"][:,2]

        dudt = torch.autograd.grad(u.sum(), data[f"t_interior_{train_or_test}_tensor"], retain_graph=True, create_graph=True)[0]
        dudx = torch.autograd.grad(u.sum(), data[f"x_interior_{train_or_test}_tensor"], retain_graph=True, create_graph=True)[0]
        dudy = torch.autograd.grad(u.sum(), data[f"y_interior_{train_or_test}_tensor"], retain_graph=True, create_graph=True)[0]
        dvdt = torch.autograd.grad(v.sum(), data[f"t_interior_{train_or_test}_tensor"], retain_graph=True, create_graph=True)[0]
        dvdx = torch.autograd.grad(v.sum(), data[f"x_interior_{train_or_test}_tensor"], retain_graph=True, create_graph=True)[0]
        dvdy = torch.autograd.grad(v.sum(), data[f"y_interior_{train_or_test}_tensor"], retain_graph=True, create_graph=True)[0]

        dpdx = torch.autograd.grad(p.sum(), data[f"x_interior_{train_or_test}_tensor"], retain_graph=True, create_graph=True)[0]
        dpdy = torch.autograd.grad(p.sum(), data[f"y_interior_{train_or_test}_tensor"], retain_graph=True, create_graph=True)[0]

        d2udx2 = torch.autograd.grad(dudx.sum(), data[f"x_interior_{train_or_test}_tensor"], retain_graph=True, create_graph=True)[0]
        d2udy2 = torch.autograd.grad(dudy.sum(), data[f"y_interior_{train_or_test}_tensor"], retain_graph=True, create_graph=True)[0]
        d2vdx2 = torch.autograd.grad(dvdx.sum(), data[f"x_interior_{train_or_test}_tensor"], retain_graph=True, create_graph=True)[0]
        d2vdy2 = torch.autograd.grad(dvdy.sum(), data[f"y_interior_{train_or_test}_tensor"], retain_graph=True, create_graph=True)[0]

        #Compute PDE value for all X,Y,T locations in training set
        #according to derivates and constants
        pde_u = dudt + (u * dudx) + (v * dudy) - (d2udx2 + d2udy2) + dpdx
        pde_v = dvdt + (u * dvdx) + (v * dvdy) - (d2vdx2 + d2vdy2) + dpdy
        pde_conserve_mass = dudx + dvdy

        #Define loss criterion
        criterion = self.criterion

        #Both components of the PDE should minimize to zero so creating
        #a target array equal to zero
        pde_loss_u = criterion(pde_u,torch.zeros_like(pde_u))
        pde_loss_v = criterion(pde_v,torch.zeros_like(pde_v))
        pde_loss_mass = criterion(pde_conserve_mass,torch.zeros_like(pde_conserve_mass))
        pde_loss_total = pde_loss_u + pde_loss_v + pde_loss_mass

        data[f"{train_or_test}_pde_loss"] = pde_loss_total

        #Compute boundary loss and data residual
        data[f"{train_or_test}_boundary_loss"] = \
            criterion(
                data[f"{train_or_test}_boundary_labels"],
                data[f"{train_or_test}_boundary_pred"])

        data[f"{train_or_test}_data_loss"] = \
            criterion(
                data[f"{train_or_test}_interior_labels"],
                data[f"{train_or_test}_interior_pred"])

        return data

    def forward(self, x, y, t):
        """
        The forward pass of the CFDPINN.
        Predict fluid properties from X,Y,T inputs.
        """
        inputs = torch.cat(
            [
                x.reshape(-1,1),
                y.reshape(-1,1),
                t.reshape(-1,1)
            ],
            axis=1)

        return self.linear_stack(inputs)

    @function_timer
    def train(self,data,args):
        """
        The control or logic loop for training the CFDPINN.
        """
        data["train_data_loss_weight_prev"] = 0
        data["train_pde_loss_weight_prev"] = 0
        data["train_boundary_loss_weight_prev"] = 0

        if args.adaption == "softadapt":
            data[f"train_data_losses"] = []
            data[f"train_pde_losses"] = []
            data[f"train_boundary_losses"] = []
        
        if args.profile:
            prof = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(args.profile_path),
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
                with_flops=True)

            prof.start()

        for epoch in tqdm(range(1,self.epochs + 1), desc="CFD PINN training progress"):

            if args.profile:
                self.train_loop(data,epoch)
                prof.step()

            else:
                self.train_loop(data,epoch)
                self.test_loop(data,epoch)

        if args.profile:
            prof.stop()

        if self.writer:
            self.writer.close()

    def train_loop(self,data,epoch):
        """
        The function called during each epoch of
        CFDPINN training
        """
        #Ensure gradients are set to zero
        self.optimizer.zero_grad()

        #Get predictions and labels for fluid properties
        data_labels = ["interior","boundary"]
        for data_label in data_labels:
            data[f"train_{data_label}_pred"] = self.forward(
                data[f"x_{data_label}_train_tensor"],
                data[f"y_{data_label}_train_tensor"],
                data[f"t_{data_label}_train_tensor"]
            )

            data[f"train_{data_label}_labels"] = torch.cat(
                [
                    data[f"u_{data_label}_train_tensor"].reshape(-1,1),
                    data[f"v_{data_label}_train_tensor"].reshape(-1,1),
                    data[f"p_{data_label}_train_tensor"].reshape(-1,1)
                ],axis=1
            )

        #Get loss for u,v and p on interior, PDE and boundary conditions
        data = self.lossfn(data,"train")

        #Adapt weights
        data = self.weight_adaption(data,epoch)

        #Create a weighted total loss to update all network parameters
        data["train_weighted_total_loss"] = \
            (data["train_data_loss"] * data["train_data_loss_weight"]) + \
            (data["train_pde_loss_weight"] * data["train_pde_loss"]) + \
            (data["train_boundary_loss_weight"] * data["train_boundary_loss"])

        #Write QC data to Tensorboard
        if self.writer:
            self.tensorboard_outputs(data,epoch,"train")

        #Update weights according to weighted loss function
        data["train_weighted_total_loss"].backward()
        self.optimizer.step()

    def weight_adaption(self,data,epoch):
        """
        Apply the user chosen weighting methodology to 
        the various loss tensors in the multi-objective function.
        """
        if self.adaption == "softadapt":
            data = self.apply_softadapt(data,epoch)
        
        elif self.adaption == "lrannealing":
            data = self.apply_lrannealing(data,epoch)
        
        elif self.adaption == "noadaption":
            data = self.dummy_adaption(data,epoch)
        
        else:
            msg = f"Somehow you have managed to pass an --adaption name that is not supported"
            raise Exception(msg)

        return data

    def apply_softadapt(self,data,epoch):
        """
        Implement the SoftAdapt weighting algorithm.

        Uses the library softadapt https://github.com/dr-aheydari/SoftAdapt
        which implements the algorithm described in the paper below.

        @article{DBLP:journals/corr/abs-1912-12355,
            author    = {A. Ali Heydari and
                    Craig A. Thompson and
                    Asif Mehmood},
            title     = {SoftAdapt: Techniques for Adaptive Loss Weighting of Neural Networks
                    with Multi-Part Loss Functions},
            journal   = {CoRR},
            volume    = {abs/1912.12355},
            year      = {2019},
            url       = {http://arxiv.org/abs/1912.12355},
            eprinttype = {arXiv},
            eprint    = {1912.12355},
            timestamp = {Fri, 03 Jan 2020 16:10:45 +0100},
            biburl    = {https://dblp.org/rec/journals/corr/abs-1912-12355.bib},
            bibsource = {dblp computer science bibliography, https://dblp.org}
        }
        """
        adapt_weights = torch.tensor([1,1,1])
        loss_weighted_softadapt_object  = LossWeightedSoftAdapt(beta=0.1)

        data[f"train_data_losses"].append(data["train_data_loss"])
        data[f"train_pde_losses"].append(data["train_pde_loss"])
        data[f"train_boundary_losses"].append(data["train_boundary_loss"])

        if epoch > 5:
            adapt_weights = loss_weighted_softadapt_object.get_component_weights(
                torch.tensor(data[f"train_data_losses"][epoch-5:epoch]),
                torch.tensor(data[f"train_pde_losses"][epoch-5:epoch]),
                torch.tensor(data[f"train_boundary_losses"][epoch-5:epoch]),
                verbose=False)

        #Create total loss to update all network parameters
        data["train_data_loss_weight"] = adapt_weights[0]
        data["train_pde_loss_weight"] = adapt_weights[1]
        data["train_boundary_loss_weight"] = adapt_weights[2]
        
        return data
    
    def apply_lrannealing(self,data,epoch):
        """
        Applies the Learning rate annealing algorithm by hand.

        As described by Wang et al. 2020 in the paper below:

        Wang, S., Teng, Y., and Perdikaris, P. 
        Understanding and mitigating gradient pathologies in 
        physics-informed neural networks. arXiv e-prints (Jan. 2020), 
        arXiv:2001.04536.        
        """
        #Calculate dBoundaryLoss/dTheta
        data_labels = ["boundary","pde","data"]
        for data_label in data_labels:
            data[f"train_{data_label}_loss"].backward(retain_graph=True)
            data[f"{data_label}_grads"] = []
            data[f"{data_label}_grads_mean"] = []
            data[f"{data_label}_grads_max"] = []

            for name, param in self.named_parameters():
                if "weight" in name:
                    data[f"{data_label}_grads_mean"].append(torch.mean(torch.abs(param.grad.view(-1))))
                    data[f"{data_label}_grads_max"].append(torch.max(torch.abs(param.grad.view(-1))))

            self.optimizer.zero_grad()

        #Compute adaptive weight for each component of total loss
        #relative to the mean gradient of data_loss w.r.t layer weights
        data["train_boundary_loss_weight"] = \
            torch.max(torch.tensor(data["data_grads_max"])) / \
            torch.mean(torch.tensor(data["boundary_grads_mean"]))

        data["train_pde_loss_weight"] = \
            torch.max(torch.tensor(data["data_grads_max"])) / \
            torch.mean(torch.tensor(data["pde_grads_mean"]))
        
        data["train_data_loss_weight"] = \
            torch.max(torch.tensor(data["data_grads_max"])) / \
            torch.mean(torch.tensor(data["data_grads_mean"]))

        data["train_data_loss_weight"] = data["train_data_loss_weight"]*self.alpha + (1-self.alpha)*data["train_data_loss_weight_prev"]
        data["train_data_loss_weight_prev"] = data["train_data_loss_weight"]

        data["train_pde_loss_weight"] = data["train_pde_loss_weight"]*self.alpha + (1-self.alpha)*data["train_pde_loss_weight_prev"]
        data["train_pde_loss_weight_prev"] = data["train_pde_loss_weight"]

        data["train_boundary_loss_weight"] = data["train_boundary_loss_weight"]*self.alpha + (1-self.alpha)*data["train_boundary_loss_weight_prev"]
        data["train_boundary_loss_weight_prev"] = data["train_boundary_loss_weight"]

        return data
    
    def dummy_adaption(self,data,epoch):
        """
        Equally weight all loss components
        """
        data["train_boundary_loss_weight"] = 1
        data["train_pde_loss_weight"] = 1
        data["train_data_loss_weight"] = 1
        
        return data

    def test_loop(self,data,epoch):
        """
        The function called during each testing phase
        of the training loop.
        """
        data_labels = ["interior","boundary"]
        for data_label in data_labels:
            data[f"test_{data_label}_pred"] = self.forward(
                data[f"x_{data_label}_test_tensor"],
                data[f"y_{data_label}_test_tensor"],
                data[f"t_{data_label}_test_tensor"]
            )

            #Concatenate u,v,p into single tensor for loss calculation
            data[f"test_{data_label}_labels"] = torch.cat(
                [
                    data[f"u_{data_label}_test_tensor"].reshape(-1,1),
                    data[f"v_{data_label}_test_tensor"].reshape(-1,1),
                    data[f"p_{data_label}_test_tensor"].reshape(-1,1)
                ],axis=1
            )

        #Get loss for u,v and p on interior, PDE and boundary conditions
        data = self.lossfn(data,"test")

        #Calculate a total loss
        data["test_total_loss"] = \
            data["test_data_loss"] + data["test_pde_loss"] + data["test_boundary_loss"]

        #Write QC data to Tensorboard if enabled
        if self.writer:
            self.tensorboard_outputs(data,epoch,"test")

    def tensorboard_outputs(self,data,epoch,train_or_test):
        """
        Write QC data to Tensorboard for real-time and
        post-mortem analysis
        """
        if train_or_test == "train":
            #Raw losses
            self.writer.add_scalar('train_boundary_loss',data["train_boundary_loss"],epoch)
            self.writer.add_scalar('train_pde_loss',data["train_pde_loss"],epoch)
            self.writer.add_scalar('train_data_loss',data["train_data_loss"],epoch)

            #Weighted losses
            self.writer.add_scalar('train_weighted_boundary_loss',data["train_boundary_loss"] * data["train_boundary_loss_weight"],epoch)
            self.writer.add_scalar('train_weighted_pde_loss',data["train_pde_loss"] * data["train_pde_loss_weight"],epoch)
            self.writer.add_scalar('train_weighted_data_loss',data["train_data_loss"] * data["train_data_loss_weight"],epoch)
            self.writer.add_scalar('train_weighted_total_loss',data["train_weighted_total_loss"],epoch)

            #Adaptive loss weights
            self.writer.add_scalar('train_boundary_loss_weight',data["train_boundary_loss_weight"],epoch)
            self.writer.add_scalar('train_pde_loss_weight',data["train_pde_loss_weight"],epoch)
            self.writer.add_scalar('train_data_loss_weight',data["train_data_loss_weight"],epoch)

        elif train_or_test == "test":
            #Raw losses
            self.writer.add_scalar('test_boundary_loss',data["test_boundary_loss"],epoch)
            self.writer.add_scalar('test_pde_loss',data["test_pde_loss"],epoch)
            self.writer.add_scalar('test_data_loss',data["test_data_loss"],epoch)
            self.writer.add_scalar('test_total_loss',data["test_total_loss"],epoch)

def save_model(pinn):
    """
    Write the CFDPINN model to disk for later use.
    """
    print(f"Saving CFDPINN model to {pinn.model_output_path}...")
    torch.save(pinn.to("cpu"), pinn.model_output_path)
    print(f"\tModel {pinn.model_output_path} saved\n")

def predict_fluid(data,pinn,geom,args):
    """
    A high-level function to conduct prediction of fluid 
    properties using a trained CFDPINN model.

    This includes pre-processing and post-processing
    """
    print("Prediction of fluid properties...")

    if args.debug:
        print("\tDEBUG: Applying scaler...")
    scaled_features = data["scaler"].transform(data["features"])
    if args.debug:
        print("\tDEBUG: Scaler applied")

    t = torch.from_numpy(scaled_features[:,0]).float().to(pinn.device)
    y = torch.from_numpy(scaled_features[:,1]).float().to(pinn.device)
    x = torch.from_numpy(scaled_features[:,2]).float().to(pinn.device)

    if args.debug:
        print(f"\tDEBUG: len(x): {len(x)}")
        print(f"\tDEBUG: len(y): {len(y)}")
        print(f"\tDEBUG: len(t): {len(t)}")

    prediction = inference(pinn,args,x,y,t)
    prediction = prediction.cpu().detach().numpy()

    #Get fluid dynamics components out of prediction
    #and shape into same dimensions as the loaded simulation
    #from openfoam numpy array
    shape = (
        geom["numt"],
        geom["numy"],
        geom["numx"])

    data["u_pred"] = prediction[:,0].reshape(shape)
    data["v_pred"] = prediction[:,1].reshape(shape)
    data["p_pred"] = prediction[:,2].reshape(shape)

    print("\tFluid properties prediction complete\n")

    return data

@function_timer
def inference(pinn,args,x,y,t):
    """
    A function to run the forward process of the CFDPINN model
    Including logic for accurate timing information recording on
    both CPU and GPUs.
    """
    # If we are timing inference data we must
    # take into consideration the GPU initialization time
    # which can skew results.

    # To overcome this we will have two code paths. One for timing
    # and the other for simple doing the inferece.

    # In the timing code path, we will manually warm up the GPU
    # before recording the inference time for a fixed number of
    # repetitions.

    # A min,max,median and std.dev will then be reported for inference.
    # The same approach can be used on the CPU even though it may be
    # slightly over the top as it will still warm the cache.

    if args.inference_timing:
        #Warm up GPU
        for _ in range(5):
            pinn(x,y,t)

        #Record inference times
        repetitions = 10
        timings=zeros((repetitions,1))

        if pinn.device == "cuda":
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)

            with torch.inference_mode():
                #Get stats over 100 inferencing iterations
                for repetition in range(repetitions):
                    starter.record()
                    _ = pinn(x,y,t)
                    ender.record()
                    torch.cuda.synchronize()
                    #Beware torch.cuda.Event using milliseconds
                    #Conversion using * 0.001 to seconds is required
                    curr_time = starter.elapsed_time(ender) * 0.001
                    timings[repetition] = curr_time

        else:
            with torch.inference_mode():
                #Get stats over multiple inference iterations
                for repetition in range(repetitions):
                    start = time()
                    _ = pinn(x,y,t)
                    end = time()
                    curr_time = end - start
                    timings[repetition] = curr_time

        print("\n** Inference Timings **\n")
        print(f"DEVICE: {pinn.device}")
        print(f"\tMedian: {median(timings)}")
        print(f"\tMin: {min(timings)}")
        print(f"\tMax: {max(timings)}")
        print(f"\tStd.dev: {std(timings)}\n")

    if args.profile:
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=0,warmup=0,active=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(args.profile_path),
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
            with_flops=True)

        prof.start()
        with torch.inference_mode():
            prediction = pinn(x,y,t)
        prof.stop()

    else:
        with torch.inference_mode():
            prediction = pinn(x,y,t)

    return prediction

def compute_residual(data):
    """
    Compute the residual data between simulated
    and predicted fluid properties.
    """
    data["u_residual"] = absolute(data["u"] - data["u_pred"])
    data["v_residual"] = absolute(data["v"] - data["v_pred"])
    data["p_residual"] = absolute(data["p"] - data["p_pred"])

    return data

def load_pinn(args):
    """
    Load a CFDPINN model from disk.
    """
    pinn = torch.load(args.load_model_path)
    pinn.choose_model_device(args)
    return pinn
