#!/usr/bin/env python3 

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math

def create_animation(data,geom,num_frames,array_label):
    """
    """
    fig, ax = plt.subplots(2,2, figsize=(15, 12))

    if array_label == "train":
        label = ""
    elif array_label == "pred":
        label = "_pred"
    elif array_label == "residual":
        label = "_residual"

    print(f"Creating animation {label}...")

    #Compute magnitude of velocity from prediction
    data[f"u_mag{label}"] = np.sqrt(
            data[f"u{label}"].astype(np.double)**2 + \
            data[f"v{label}"].astype(np.double)**2
        ).astype(float)

    #Use first timestep from real simulation after initial condition to set the colour bar
    fig.colorbar(ax[0][0].pcolormesh(
        geom["grid2d_x"],geom["grid2d_x"], data[f"u"][1,:,:], 
        shading='gouraud'), 
        shrink=0.6, 
        orientation='vertical')

    fig.colorbar(ax[0][1].pcolormesh(
        geom["grid2d_x"],geom["grid2d_x"], data[f"v"][1,:,:], 
        shading='gouraud'), 
        shrink=0.6, 
        orientation='vertical')
    
    fig.colorbar(ax[1][0].pcolormesh(
        geom["grid2d_x"],geom["grid2d_x"], data[f"p"][1,:,:], 
        shading='gouraud'), 
        shrink=0.6, 
        orientation='vertical')
    
    fig.colorbar(ax[1][1].pcolormesh(
        geom["grid2d_x"],geom["grid2d_x"], data[f"u_mag{label}"][1,:,:], 
        shading='gouraud'), 
        shrink=0.6, 
        orientation='vertical')
    
    writervideo = animation.FFMpegWriter(fps=20)
    output_path = (f"./animations/tmp{label}.mp4")
    
    fluid_animation = \
        animation.FuncAnimation(
            fig, 
            animate_subplot,
            num_frames,
            fargs=(ax,data,geom,label,num_frames), 
            interval=1,
            blit=False)
    
    fig.tight_layout(pad=5)
    
    fluid_animation.save(output_path,writer=writervideo)

    print(f"\tAnimation {label} completed\n")

def animate_subplot(i,ax,data,geom,label,num_frames):
    """
    """ 
    #Indexing values for plotting and GIF creation
    animation_num_frames = num_frames
    animation_write_interval = int(math.floor(geom["numt"] / animation_num_frames))
    timestep_index = animation_write_interval * i
    timestep_time = timestep_index * geom["t_dt"]

    ax[1][0].text(0.05,0.05,f"Timestep: {timestep_time}",bbox={'facecolor':'w', 'alpha':0.5})

    ax[0][0].pcolormesh(
        geom["grid2d_x"],geom["grid2d_y"],
        data[f"u{label}"][timestep_index,:,:],
        shading='gouraud')

    if timestep_index != 0:
        ax[0][0].quiver(
            geom["grid2d_x"],geom["grid2d_y"], 
            data[f"u{label}"][timestep_index,:,:],
            data[f"v{label}"][timestep_index,:,:])

    ax[0][0].scatter(
        data[f"interior_training_locs"][data[f"interior_training_locs"][:,0] == timestep_time][:,2],
        data[f"interior_training_locs"][data[f"interior_training_locs"][:,0] == timestep_time][:,1], 
        c="Red",
        alpha=1,
        marker="x")
    
    ax[0][0].scatter(
        data["rightwall_training_locs"][data["rightwall_training_locs"][:,0] == timestep_time][:,2],
        data["rightwall_training_locs"][data["rightwall_training_locs"][:,0] == timestep_time][:,1], 
        c="Purple",
        alpha=1,
        marker="x")
    
    ax[0][0].scatter(
        data["leftwall_training_locs"][data["leftwall_training_locs"][:,0] == timestep_time][:,2],
        data["leftwall_training_locs"][data["leftwall_training_locs"][:,0] == timestep_time][:,1], 
        c="Purple",
        alpha=1,
        marker="x")
    
    ax[0][0].scatter(
        data["basewall_training_locs"][data["basewall_training_locs"][:,0] == timestep_time][:,2],
        data["basewall_training_locs"][data["basewall_training_locs"][:,0] == timestep_time][:,1], 
        c="Yellow",
        alpha=1,
        marker="x")

    ax[0][0].set_title("Velocity (u)")
    ax[0][0].set_ylabel("Spatial y index")
    ax[0][0].set_xlabel("Spatial x index")
    ax[0][0].set_box_aspect(1)

    ax[0][1].pcolormesh(
        geom["grid2d_x"],geom["grid2d_y"],
        data[f"v{label}"][timestep_index,:,:],
        shading='gouraud')
    
    if timestep_index != 0:
        ax[0][1].quiver(
            geom["grid2d_x"],geom["grid2d_y"],
            data[f"u{label}"][timestep_index,:,:],
            data[f"v{label}"][timestep_index,:,:])

    ax[0][1].scatter(
        data["interior_training_locs"][data["interior_training_locs"][:,0] == timestep_time][:,2],
        data["interior_training_locs"][data["interior_training_locs"][:,0] == timestep_time][:,1], 
        c="Red",
        alpha=1,
        marker="x")
    
    ax[0][1].scatter(
        data["rightwall_training_locs"][data["rightwall_training_locs"][:,0] == timestep_time][:,2],
        data["rightwall_training_locs"][data["rightwall_training_locs"][:,0] == timestep_time][:,1], 
        c="Purple",
        alpha=1,
        marker="x")
    
    ax[0][1].scatter(
        data["leftwall_training_locs"][data["leftwall_training_locs"][:,0] == timestep_time][:,2],
        data["leftwall_training_locs"][data["leftwall_training_locs"][:,0] == timestep_time][:,1], 
        c="Purple",
        alpha=1,
        marker="x")
    
    ax[0][1].scatter(
        data["basewall_training_locs"][data["basewall_training_locs"][:,0] == timestep_time][:,2],
        data["basewall_training_locs"][data["basewall_training_locs"][:,0] == timestep_time][:,1], 
        c="Yellow",
        alpha=1,
        marker="x")

    ax[0][1].set_title("Velocity (v)")
    ax[0][1].set_ylabel("Spatial y index")
    ax[0][1].set_xlabel("Spatial x index")
    ax[0][1].set_box_aspect(1)

    ax[1][0].pcolormesh(
        geom["grid2d_x"],geom["grid2d_y"],
        data[f"p{label}"][timestep_index,:,:],
        shading='gouraud')
    
    if timestep_index != 0:
        ax[1][0].quiver(
            geom["grid2d_x"],geom["grid2d_y"],
            data[f"u{label}"][timestep_index,:,:],
            data[f"v{label}"][timestep_index,:,:])

    ax[1][0].scatter(
        data["interior_training_locs"][data["interior_training_locs"][:,0] == timestep_time][:,2],
        data["interior_training_locs"][data["interior_training_locs"][:,0] == timestep_time][:,1], 
        c="Red",
        alpha=1,
        marker="x")
    
    ax[1][0].scatter(
        data["rightwall_training_locs"][data["rightwall_training_locs"][:,0] == timestep_time][:,2],
        data["rightwall_training_locs"][data["rightwall_training_locs"][:,0] == timestep_time][:,1], 
        c="Purple",
        alpha=1,
        marker="x")
    
    ax[1][0].scatter(
        data["leftwall_training_locs"][data["leftwall_training_locs"][:,0] == timestep_time][:,2],
        data["leftwall_training_locs"][data["leftwall_training_locs"][:,0] == timestep_time][:,1], 
        c="Purple",
        alpha=1,
        marker="x")
    
    ax[1][0].scatter(
        data["basewall_training_locs"][data["basewall_training_locs"][:,0] == timestep_time][:,2],
        data["basewall_training_locs"][data["basewall_training_locs"][:,0] == timestep_time][:,1], 
        c="Yellow",
        alpha=1,
        marker="x")

    ax[1][0].set_title("Pressure")
    ax[1][0].set_ylabel("Spatial y index")
    ax[1][0].set_xlabel("Spatial x index")
    ax[1][0].set_box_aspect(1)

    ax[1][1].pcolormesh(
        geom["grid2d_x"],geom["grid2d_y"],
        data[f"u_mag{label}"][timestep_index,:,:],
        shading='gouraud')
    
    if timestep_index != 0:
        ax[1][1].quiver(
            geom["grid2d_x"],geom["grid2d_y"],
            data[f"u{label}"][timestep_index,:,:],
            data[f"v{label}"][timestep_index,:,:])

    ax[1][1].scatter(
        data["interior_training_locs"][data["interior_training_locs"][:,0] == timestep_time][:,2],
        data["interior_training_locs"][data["interior_training_locs"][:,0] == timestep_time][:,1], 
        c="Red",
        alpha=1,
        marker="x")
    
    ax[1][1].scatter(
        data["rightwall_training_locs"][data["rightwall_training_locs"][:,0] == timestep_time][:,2],
        data["rightwall_training_locs"][data["rightwall_training_locs"][:,0] == timestep_time][:,1], 
        c="Purple",
        alpha=1,
        marker="x")
    
    ax[1][1].scatter(
        data["leftwall_training_locs"][data["leftwall_training_locs"][:,0] == timestep_time][:,2],
        data["leftwall_training_locs"][data["leftwall_training_locs"][:,0] == timestep_time][:,1], 
        c="Purple",
        alpha=1,
        marker="x")
    
    ax[1][1].scatter(
        data["basewall_training_locs"][data["basewall_training_locs"][:,0] == timestep_time][:,2],
        data["basewall_training_locs"][data["basewall_training_locs"][:,0] == timestep_time][:,1], 
        c="Yellow",
        alpha=1,
        marker="x")

    ax[1][1].set_title("Velocity magntiude")
    ax[1][1].set_ylabel("Spatial y index")
    ax[1][1].set_xlabel("Spatial x index")
    ax[1][1].set_box_aspect(1)

def static_plots(data,args,geom):
    """
    """
    return 0
