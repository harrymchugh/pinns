#!/usr/bin/env python3 

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math

def create_animation(data,geom,args,train_pred):
    """
    """
    animation = animation.FuncAnimation(fig, animate_subplot, int(animation_num_frames), interval=1)
    writervideo = animation.FFMpegWriter(fps=20)
    output_path = f'../../animations/full_openfoam_ofeqns_prediction_outputframes{animation_num_frames}.mp4'
    animation.save(output_path,writer=writervideo)

def animation_initialisation(data,geom,args,train_pred):
    if train_pred == "train":
        label = ""
    elif train_pred == "pred":
        label = "_pred"

    animation_num_frames = 50
    animation_write_interval = int(math.floor(geom["numt"] / animation_num_frames))

    #Setup figure
    fig, ax = plt.subplots(2,2, figsize=(12, 12))
    fig.tight_layout()

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

def animate_subplot(i):
        #Indexing values for plotting and GIF creation
        timestep_index = animation_write_interval * i
        timestep_time = timestep_index * dt

        fig.suptitle(f"Timestep value: {timestep_time}", fontsize=16)

        ax[0][0].pcolormesh(grid2d_x,grid2d_y, u_pred[timestep_index,:,:], shading='gouraud')

        if timestep_index != 0:
            ax[0][0].quiver(grid2d_x,grid2d_y, u_pred[timestep_index,:,:],v_pred[timestep_index,:,:])

        ax[0][0].scatter(
            interior_training_locs[interior_training_locs[:,0] == timestep_time][:,2],
            interior_training_locs[interior_training_locs[:,0] == timestep_time][:,1], 
            c="Red",
            alpha=1,
            marker="x")
        ax[0][0].scatter(
            rightwall_training_locs[rightwall_training_locs[:,0] == timestep_time][:,2],
            rightwall_training_locs[rightwall_training_locs[:,0] == timestep_time][:,1], 
            c="Purple",
            alpha=1,
            marker="x")
        ax[0][0].scatter(
            leftwall_training_locs[leftwall_training_locs[:,0] == timestep_time][:,2],
            leftwall_training_locs[leftwall_training_locs[:,0] == timestep_time][:,1], 
            c="Purple",
            alpha=1,
            marker="x")
        ax[0][0].scatter(
            basewall_training_locs[basewall_training_locs[:,0] == timestep_time][:,2],
            basewall_training_locs[basewall_training_locs[:,0] == timestep_time][:,1], 
            c="Yellow",
            alpha=1,
            marker="x")

        ax[0][0].set_title("Velocity (u)")
        ax[0][0].set_ylabel("Spatial y index")
        ax[0][0].set_xlabel("Spatial x index")
        ax[0][0].set_box_aspect(1)

        ax[0][1].pcolormesh(grid2d_x,grid2d_y, v_pred[timestep_index,:,:], shading='gouraud')
        if timestep_index != 0:
            ax[0][1].quiver(grid2d_x,grid2d_y, u_pred[timestep_index,:,:],v_pred[timestep_index,:,:])

        ax[0][1].scatter(
            interior_training_locs[interior_training_locs[:,0] == timestep_time][:,2],
            interior_training_locs[interior_training_locs[:,0] == timestep_time][:,1], 
            c="Red",
            alpha=1,
            marker="x")
        ax[0][1].scatter(
            rightwall_training_locs[rightwall_training_locs[:,0] == timestep_time][:,2],
            rightwall_training_locs[rightwall_training_locs[:,0] == timestep_time][:,1], 
            c="Purple",
            alpha=1,
            marker="x")
        ax[0][1].scatter(
            leftwall_training_locs[leftwall_training_locs[:,0] == timestep_time][:,2],
            leftwall_training_locs[leftwall_training_locs[:,0] == timestep_time][:,1], 
            c="Purple",
            alpha=1,
            marker="x")
        ax[0][1].scatter(
            basewall_training_locs[basewall_training_locs[:,0] == timestep_time][:,2],
            basewall_training_locs[basewall_training_locs[:,0] == timestep_time][:,1], 
            c="Yellow",
            alpha=1,
            marker="x")

        ax[0][1].set_title("Velocity (v)")
        ax[0][1].set_ylabel("Spatial y index")
        ax[0][1].set_xlabel("Spatial x index")
        ax[0][1].set_box_aspect(1)

        ax[1][0].pcolormesh(grid2d_x,grid2d_y, p_pred[timestep_index,:,:], shading='gouraud')
        if timestep_index != 0:
            ax[1][0].quiver(grid2d_x,grid2d_y, u_pred[timestep_index,:,:],v_pred[timestep_index,:,:])

        ax[1][0].scatter(
            interior_training_locs[interior_training_locs[:,0] == timestep_time][:,2],
            interior_training_locs[interior_training_locs[:,0] == timestep_time][:,1], 
            c="Red",
            alpha=1,
            marker="x")
        ax[1][0].scatter(
            rightwall_training_locs[rightwall_training_locs[:,0] == timestep_time][:,2],
            rightwall_training_locs[rightwall_training_locs[:,0] == timestep_time][:,1], 
            c="Purple",
            alpha=1,
            marker="x")
        ax[1][0].scatter(
            leftwall_training_locs[leftwall_training_locs[:,0] == timestep_time][:,2],
            leftwall_training_locs[leftwall_training_locs[:,0] == timestep_time][:,1], 
            c="Purple",
            alpha=1,
            marker="x")
        ax[1][0].scatter(
            basewall_training_locs[basewall_training_locs[:,0] == timestep_time][:,2],
            basewall_training_locs[basewall_training_locs[:,0] == timestep_time][:,1], 
            c="Yellow",
            alpha=1,
            marker="x")

        ax[1][0].set_title("Pressure")
        ax[1][0].set_ylabel("Spatial y index")
        ax[1][0].set_xlabel("Spatial x index")
        ax[1][0].set_box_aspect(1)

        ax[1][1].pcolormesh(grid2d_x,grid2d_y, u_mag_pred[timestep_index,:,:], shading='gouraud')
        if timestep_index != 0:
            ax[1][1].quiver(grid2d_x,grid2d_y, u_pred[timestep_index,:,:],v_pred[timestep_index,:,:])

        ax[1][1].scatter(
            interior_training_locs[interior_training_locs[:,0] == timestep_time][:,2],
            interior_training_locs[interior_training_locs[:,0] == timestep_time][:,1], 
            c="Red",
            alpha=1,
            marker="x")
        ax[1][1].scatter(
            rightwall_training_locs[rightwall_training_locs[:,0] == timestep_time][:,2],
            rightwall_training_locs[rightwall_training_locs[:,0] == timestep_time][:,1], 
            c="Purple",
            alpha=1,
            marker="x")
        ax[1][1].scatter(
            leftwall_training_locs[leftwall_training_locs[:,0] == timestep_time][:,2],
            leftwall_training_locs[leftwall_training_locs[:,0] == timestep_time][:,1], 
            c="Purple",
            alpha=1,
            marker="x")
        ax[1][1].scatter(
            basewall_training_locs[basewall_training_locs[:,0] == timestep_time][:,2],
            basewall_training_locs[basewall_training_locs[:,0] == timestep_time][:,1], 
            c="Yellow",
            alpha=1,
            marker="x")

        ax[1][1].set_title("Velocity magntiude")
        ax[1][1].set_ylabel("Spatial y index")
        ax[1][1].set_xlabel("Spatial x index")
        ax[1][1].set_box_aspect(1)

def static_plots():
    """
    """
    return 0
