#!/usr/bin/env python3

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch import from_numpy
from cfdpinn.plots import create_animation

def preprocess(data,geom,args):
    """
    Returns a dict holding the NumPy arrays needed for PINN training
    """
    #Merge features x,y,t into single array in data
    data = merge_features(data,geom)

    #Seperate into boundary and interior arrays
    data = extract_boundaries(data)
    data = extract_interior(data)

    #Obtain feature scaling object
    #Note that scaling is not applied until after training 
    #locations are obtained as this requires using 
    #pre-scaled spatio-temporal locations
    data["scaler"] = scaling_object(data)

    #Create training locations
    training_locations = get_training_locations(data,args)

    #Train-test-splitting
    data = apply_scaling(data)
    data = apply_train_test_split(data,args.test_size,scaled=True)

    #Create animation if needed
    if args.training_animation:
        create_animation(data,training_locations)

    #Make boundary arrays contiguous for DL framework
    data = make_boundary_arrays_contiguous(data)

    return data

def scaling_object(data):
    """
    """
    _scaler = StandardScaler()
    scaler = _scaler.fit(data["features"])
    
    return scaler

def apply_scaling(data):
    """
    """
    data_labels = ["basewall","interior","leftwall","rightwall"]
    for data_label in data_labels:
        data[f"scaled_features_{data_label}"] = \
            data["scaler"].transform(data[f"features_{data_label}"])
    
    return data

def get_training_locations(data,args):
    """
    """
    #First need to apply train_test_splitting to 
    #replicate real train_test_split
    _data = apply_train_test_split(data,args.test_size,scaled=False)

    #Now extract locations for all components of
    #training data arrays
    array_labels = ["interior","basewall","rightwall","leftwall"]
    for array_label in array_labels:
        _data[f"{array_label}_training_locs"] = np.concatenate(
            (
                _data[f"t_{array_label}_train"].flatten().reshape(-1,1),
                _data[f"y_{array_label}_train"].flatten().reshape(-1,1),
                _data[f"x_{array_label}_train"].flatten().reshape(-1,1)
            ), 
            axis=1)

    _data[f"{array_label}_training_locs"] = \
        _data[f"{array_label}_training_locs"][_data[f"{array_label}_training_locs"][:,0].argsort()]

    return _data

def apply_train_test_split(data,test_size,scaled):
    """
    """
    if scaled == True:
        label = "scaled_"
    elif scaled == False:
        label = ""

    #Interior
    (
        data["x_interior_train"],
        data["x_interior_test"],
        data["y_interior_train"],
        data["y_interior_test"],
        data["t_interior_train"],
        data["t_interior_test"],
        data["u_interior_train"],
        data["u_interior_test"],
        data["v_interior_train"],
        data["v_interior_test"],
        data["p_interior_train"],
        data["p_interior_test"],
    ) = train_test_split(
        data[f"{label}features_interior"][:,2], 
        data[f"{label}features_interior"][:,1], 
        data[f"{label}features_interior"][:,0],
        data["u_interior_labels"],
        data["v_interior_labels"],
        data["p_interior_labels"],
        test_size=test_size)

    #Basewall
    (
        data["x_basewall_train"],
        data["x_basewall_test"],
        data["y_basewall_train"],
        data["y_basewall_test"],
        data["t_basewall_train"],
        data["t_basewall_test"],
        data["u_basewall_train"],
        data["u_basewall_test"],
        data["v_basewall_train"],
        data["v_basewall_test"],
        data["p_basewall_train"],
        data["p_basewall_test"],
    ) = train_test_split(
        data[f"{label}features_basewall"][:,2], 
        data[f"{label}features_basewall"][:,1], 
        data[f"{label}features_basewall"][:,0],
        data["u_basewall_labels"],
        data["v_basewall_labels"],
        data["p_basewall_labels"],
        test_size=test_size)

    #Leftwall
    (
        data["x_leftwall_train"],
        data["x_leftwall_test"],
        data["y_leftwall_train"],
        data["y_leftwall_test"],
        data["t_leftwall_train"],
        data["t_leftwall_test"],
        data["u_leftwall_train"],
        data["u_leftwall_test"],
        data["v_leftwall_train"],
        data["v_leftwall_test"],
        data["p_leftwall_train"],
        data["p_leftwall_test"],
    ) = train_test_split(
        data[f"{label}features_leftwall"][:,2], 
        data[f"{label}features_leftwall"][:,1], 
        data[f"{label}features_leftwall"][:,0],
        data["u_leftwall_labels"],
        data["v_leftwall_labels"],
        data["p_leftwall_labels"],
        test_size=test_size)

    #Rightwall
    (
        data["x_rightwall_train"],
        data["x_rightwall_test"],
        data["y_rightwall_train"],
        data["y_rightwall_test"],
        data["t_rightwall_train"],
        data["t_rightwall_test"],
        data["u_rightwall_train"],
        data["u_rightwall_test"],
        data["v_rightwall_train"],
        data["v_rightwall_test"],
        data["p_rightwall_train"],
        data["p_rightwall_test"],
    ) = train_test_split(
        data[f"{label}features_rightwall"][:,2], 
        data[f"{label}features_rightwall"][:,1], 
        data[f"{label}features_rightwall"][:,0],
        data["u_rightwall_labels"],
        data["v_rightwall_labels"],
        data["p_rightwall_labels"],
        test_size=test_size)

    return data

def make_boundary_arrays_contiguous(data):
    """
    """
    data_components = ["u","v","p","x","y","t"]
    train_test_components = ["train","test"]

    for data_component in data_components:
        for train_test_component in train_test_components:
            data[f"{data_component}_boundary_{train_test_component}"] = \
                np.concatenate((
                    data[f"{data_component}_rightwall_{train_test_component}"],
                    data[f"{data_component}_leftwall_{train_test_component}"],
                    data[f"{data_component}_basewall_{train_test_component}"]
            ))

    return data

def extract_boundaries(data):
    """
    """
    #Handling data labels; fluid properties
    array_labels = ["u", "v", "p"]
    for array_label in array_labels:
        
        #Get boundary data labels
        data[f"{array_label}_basewall"]  = \
            data[f"{array_label}"][:,0,:]
        
        data[f"{array_label}_leftwall"]  = \
            data[f"{array_label}"][:,1:-1,0]
        
        data[f"{array_label}_rightwall"] = \
            data[f"{array_label}"][:,1:-1,-1]

        #Reshape to column format for DL framework
        data[f"{array_label}_basewall_labels"]  = \
            data[f"{array_label}_basewall"].flatten().reshape(-1,1)
        
        data[f"{array_label}_leftwall_labels"]  = \
            data[f"{array_label}_leftwall"].flatten().reshape(-1,1)
        
        data[f"{array_label}_rightwall_labels"] = \
            data[f"{array_label}_rightwall"].flatten().reshape(-1,1)
    
    #Handling features; x,y,t spatio-temporal locations 
    array_labels = ["x","y","t"]
    for array_label in array_labels:
        
        #Get boundary data features
        data[f"basewall_features_{array_label}"] = data[array_label][:,0,:]
        data[f"leftwall_features_{array_label}"]  = data[array_label][:,1:-1,0]
        data[f"rightwall_features_{array_label}"] = data[array_label][:,1:-1,-1]

    #Reshape to column format for DL framework
    array_labels = ["basewall","rightwall","leftwall"]
    for array_label in array_labels:
        
        data[f"features_{array_label}"] = np.concatenate(
        (
            data[f"{array_label}_features_t"].flatten().reshape(-1,1),
            data[f"{array_label}_features_y"].flatten().reshape(-1,1),
            data[f"{array_label}_features_x"].flatten().reshape(-1,1)
        ), 
        axis=1)

    return data

def extract_interior(data):
    """
    """
    #Handling data labels; fluid properties
    array_labels = ["u", "v", "p"]
    for array_label in array_labels:
        data[f"{array_label}_interior"] = data[array_label][:,1:-1,1:-1]
        data[f"{array_label}_interior_labels"] = \
            data[f"{array_label}_interior"].flatten().reshape(-1,1)
    
    #Handling features; x,y,t spatio-temporal locations
    array_labels = ["x", "y", "t"]
    for array_label in array_labels:
        data[f"interior_features_{array_label}"] = data[array_label][:,1:-1,1:-1]

    #Reshape to column format for DL framework
    data["features_interior"] = np.concatenate(
    (
        data["interior_features_t"].flatten().reshape(-1,1),
        data["interior_features_y"].flatten().reshape(-1,1),
        data["interior_features_x"].flatten().reshape(-1,1),
    ),
    axis=1)
    
    return data

def merge_features(data,geom):
    """
    """
    data["y"], data["t"], data["x"] = np.meshgrid(
        np.linspace(geom["y_start"],geom["y_end"],geom["numy"]),
        np.linspace(geom["t_start"],geom["t_end"],geom["numt"]),
        np.linspace(geom["x_start"],geom["x_end"],geom["numx"]))

    #Return an array in form t, x, y
    data["features"] = np.concatenate(
        (
            data["t"].flatten().reshape(-1,1),
            data["y"].flatten().reshape(-1,1),
            data["x"].flatten().reshape(-1,1)
        ),
        axis=1)

    return data

def convert_to_tensors(data,device):
    """
    """
    geom_components = ["interior","boundary"]
    train_test_components = ["train","test"]

    for geom_component in geom_components:
        for train_test_component in train_test_components:

            data[f"x_{geom_component}_{train_test_component}_tensor"] = \
                from_numpy(data[f"x_{geom_component}_{train_test_component}"]).\
                    float().requires_grad_().to(device)

            data[f"y_{geom_component}_{train_test_component}_tensor"] = \
                from_numpy(data[f"y_{geom_component}_{train_test_component}"]).\
                    float().requires_grad_().to(device)

            data[f"t_{geom_component}_{train_test_component}_tensor"] = \
                from_numpy(data[f"t_{geom_component}_{train_test_component}"]).\
                    float().requires_grad_().to(device)

            data[f"u_{geom_component}_{train_test_component}_tensor"] = \
                from_numpy(data[f"u_{geom_component}_{train_test_component}"]).\
                    float().requires_grad_().to(device)

            data[f"v_{geom_component}_{train_test_component}_tensor"] = \
                from_numpy(data[f"v_{geom_component}_{train_test_component}"]).\
                    float().requires_grad_().to(device)

            data[f"p_{geom_component}_{train_test_component}_tensor"] = \
                from_numpy(data[f"p_{geom_component}_{train_test_component}"]).\
                    float().requires_grad_().to(device)

    return data