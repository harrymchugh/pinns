#!/usr/bin/env python3

import torch
from numpy import absolute
from tqdm import tqdm

#from torch.utils.tensorboard import SummaryWriter

class CfdPinn(torch.nn.Module):
    def __init__(self,args):
        """_summary_
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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)
        self.criterion = torch.nn.MSELoss()
        self.viscosity = args.viscosity
        self.tensorboard_dir_name = "weighting"
        
        #if args.tensorboard:
            #self.writer = SummaryWriter("runs/" + self.tensorboard_dir_name)
        
        self.epochs = args.epochs

        self.model_output_path = args.pinn_output_path

    def lossfn(self,data,train_or_test):
        """
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
        d2udy2 = torch.autograd.grad(dudx.sum(), data[f"y_interior_{train_or_test}_tensor"], retain_graph=True, create_graph=True)[0]
        d2vdx2 = torch.autograd.grad(dvdx.sum(), data[f"x_interior_{train_or_test}_tensor"], retain_graph=True, create_graph=True)[0]
        d2vdy2 = torch.autograd.grad(dvdx.sum(), data[f"y_interior_{train_or_test}_tensor"], retain_graph=True, create_graph=True)[0]
                
        #Compute PDE value for all X,Y,T locations in training set
        #according to derivates and constants
        pde_u = dudt + (u * dudx) + (v * dudy) - (self.viscosity * (d2udx2 + d2udy2)) + dpdx
        pde_v = dvdt + (u * dvdx) + (v * dvdy) - (self.viscosity * (d2vdx2 + d2vdy2)) + dpdy

        #Define loss criterion
        criterion = self.criterion
        
        #Both components of the PDE should minimize to zero so creating
        #a target array equal to zero
        pde_total = torch.cat((pde_u,pde_v))
        pde_labels = torch.zeros_like(pde_total)

        data[f"{train_or_test}_pde_loss"] = criterion(pde_labels, pde_total)
        
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
        """
        inputs = torch.cat(
            [
                x.reshape(-1,1),
                y.reshape(-1,1),
                t.reshape(-1,1)
            ],
            axis=1).to(self.device)
        
        return self.linear_stack(inputs)

    def train(self,data):
        """
        """
        for epoch in tqdm(range(1,self.epochs + 1), desc="CFD PINN training progress"):
            self.train_loop(data,epoch)
            self.test_loop(data,epoch)

    def train_loop(self,data,epoch):
        """
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
            ).to(self.device)

        #Get loss for u,v and p on interior, PDE and boundary conditions
        data = self.lossfn(data,"train")
        
        #Calculate dBoundaryLoss/dTheta
        data_labels = ["boundary","pde","data"]
        for data_label in data_labels:
            data[f"train_{data_label}_loss"].backward(retain_graph=True)
            data[f"{data_label}_grads"] = []
            for name, param in self.named_parameters():
                if "weight" in name:
                    data[f"{data_label}_grads"].append(param.grad.view(-1))
            
            data[f"{data_label}_grads"] = torch.cat(data[f"{data_label}_grads"])
            self.optimizer.zero_grad()
        
        #Compute adaptive weight for each component of total loss
        #relative to the mean gradient of data_loss w.r.t layer weights
        data["train_boundary_loss_weight"] = \
            torch.max(torch.abs(data["data_grads"])) / \
            torch.mean(torch.abs(data["boundary_grads"]))
        
        data["train_pde_loss_weight"] = \
            torch.max(torch.abs(data["data_grads"])) / \
            torch.mean(torch.abs(data["pde_grads"]))
        
        data["train_data_loss_weight"] = \
            torch.max(torch.abs(data["data_grads"])) / \
            torch.mean(torch.abs(data["data_grads"]))

        #Create a weighted total loss to update all network parameters
        data["train_weighted_total_loss"] = \
            (data["train_data_loss"] * data["train_data_loss_weight"]) + \
            (data["train_pde_loss_weight"] * data["train_pde_loss"]) + \
            (data["train_boundary_loss_weight"] * data["train_boundary_loss"])

        #Write QC data to Tensorboard
        #if self.writer:
            #self.tensorboard_outputs(self,data,epoch,"train")
            
        #Update weights according to weighted loss function    
        data["train_weighted_total_loss"].backward()
        self.optimizer.step()

    def test_loop(self,data,epoch):
        """
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
            ).to(self.device)
                    
        #Get loss for u,v and p on interior, PDE and boundary conditions
        data = self.lossfn(data,"test")
        
        #Calculate a total loss
        data["test_total_loss"] = \
            data["test_data_loss"] + data["test_pde_loss"] + data["test_boundary_loss"]
        
        #Write QC data to Tensorboard if enabled
        #if self.writer:
            #self.tensorboard_outputs(self,data,epoch,"test")
    
    def tensorboard_outputs(self,data,epoch,train_or_test):
        """
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
            self.writer.add_scalar('train_weighted_total_loss'.data["train_weighted_total_loss"],epoch)
            
            #Adaptive loss weights
            self.writer.add_scalar('train_boundary_loss_weight',data["train_boundary_loss_weight"],epoch)
            self.writer.add_scalar('train_pde_loss_weight',data["train_pde_loss_weight"],epoch)
            self.writer.add_scalar('train_data_loss_weight',data["train_data_loss_weight"],epoch)
            
            #Gradients
            self.writer.add_scalar('train_data_mean_grad', torch.mean(torch.abs(data["data_grads"])))
            self.writer.add_scalar('train_pde_mean_grad', torch.mean(torch.abs(data["pde_grads"])))
            self.writer.add_scalar('train_bound_mean_grad', torch.mean(torch.abs(data["bound_grads"])))
        
        elif train_or_test == "test":
            #Raw losses
            self.writer.add_scalar('test_boundary_loss',data["test_boundary_loss"],epoch)
            self.writer.add_scalar('test_pde_loss',data["test_pde_loss"],epoch)
            self.writer.add_scalar('test_data_loss',data["test_data_loss"],epoch)
            self.writer.add_scalar('test_total_loss',data["test_total_loss"],epoch)

    def save_model(self):
        torch.save(self.to("cpu"), self.model_output_path)


def predict_fluid(data,pinn,geom):
    """
    """
    scaled_features = data["scaler"].transform(data["features"])

    t = scaled_features[:,0]
    y = scaled_features[:,1]
    x = scaled_features[:,2]

    with torch.inference_mode():
        prediction = pinn(
            torch.Tensor(x),
            torch.Tensor(y),
            torch.Tensor(t))

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

    data["u_residual"] = absolute(data["u"] - data["u_pred"])
    data["v_residual"] = absolute(data["v"] - data["v_pred"])
    data["p_residual"] = absolute(data["p"] - data["p_pred"])

    return data