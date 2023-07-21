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
    y_index = int(np.floor(geom["numy"] * 0.8))
    y_value = geom["grid2d_y"][y_index][0]
    timestep_index = int(np.floor(geom["numt"] / 2))
    quiver_stepx = int(np.floor(geom["numx"] / 20))
    quiver_stepy = int(np.floor(geom["numy"] / 20))
    
    if quiver_stepx == 0:
        quiver_stepx = 1
    if quiver_stepy == 0:
        quiver_stepy = 1

    umin=data["u"][timestep_index,:,:].min()
    umax=data["u"][timestep_index,:,:].max()
    pmin=data["p"][timestep_index,:,:].min()
    pmax=data["p"][timestep_index,:,:].max()
    pmin=-1
    pmax=1

    fig, axs = plt.subplot_mosaic(
        [['u_mag_quiv', 'p_quiv'], 
        ['u_mag_quiv', 'p_quiv'],
        ['u_mag_quiv_pred', 'p_quiv_pred'], 
        ['u_mag_quiv_pred', 'p_quiv_pred'], 
        ['u', 'u'],
        ['v', 'v'],
        ['p', 'p']
        ],layout='tight')

    fig.set_figwidth(16)
    fig.set_figheight(16)

    #U magnitude with quiver plot
    data[f"u_mag"] = np.sqrt(
            data[f"u"].astype(np.double)**2 + \
            data[f"v"].astype(np.double)**2
        ).astype(float)

    data[f"u_mag_pred"] = np.sqrt(
            data[f"u_pred"].astype(np.double)**2 + \
            data[f"v_pred"].astype(np.double)**2
        ).astype(float)

    u_mag_quiv_pcol = axs["u_mag_quiv"].pcolormesh(
            geom["grid2d_x"],geom["grid2d_y"],
            data[f"u_mag"][timestep_index,:,:],
            shading='gouraud',
            vmin=umin, vmax=umax)

    u_mag_quiv_quiv = axs["u_mag_quiv"].quiver(
                geom["grid2d_x"][::quiver_stepx,::quiver_stepy],geom["grid2d_y"][::quiver_stepx,::quiver_stepy], 
                data[f"u"][timestep_index,:,:][::quiver_stepx,::quiver_stepy],
                data[f"v"][timestep_index,:,:][::quiver_stepx,::quiver_stepy])

    plt.colorbar(u_mag_quiv_pcol, ax=axs["u_mag_quiv"])

    #Show index used for extracting profile
    axs["u_mag_quiv"].axhline(y = y_value, color = 'r', linestyle = "solid")

    axs["u_mag_quiv"].set_ylabel("Y spatial index")
    axs["u_mag_quiv"].set_title("Velocity magnitude with stream lines")
    axs["u_mag_quiv"].text(0.05,0.05,f"Timestep: 2.5s",bbox={'facecolor':'w', 'alpha':0.5})

    #Pressure with quiver plot
    p_quiv_pcol = axs["p_quiv"].pcolormesh(
            geom["grid2d_x"],geom["grid2d_y"],
            data[f"p"][timestep_index,:,:],
            shading='gouraud',
            vmin=pmin, vmax=pmax)

    p_quiv_quiv = axs["p_quiv"].quiver(
                geom["grid2d_x"][::quiver_stepx,::quiver_stepy],geom["grid2d_y"][::quiver_stepx,::quiver_stepy], 
                data[f"u"][timestep_index,:,:][::quiver_stepx,::quiver_stepy],
                data[f"v"][timestep_index,:,:][::quiver_stepx,::quiver_stepy])

    plt.colorbar(p_quiv_pcol, ax=axs["p_quiv"])
    axs["p_quiv"].axhline(y = y_value, color = 'r', linestyle = "solid")
    axs["p_quiv"].set_title("Pressure with velocity stream lines")

    #PREDICTIONS
    u_mag_quiv_pred_pcol = axs["u_mag_quiv_pred"].pcolormesh(
            geom["grid2d_x"],geom["grid2d_y"],
            data[f"u_mag_pred"][timestep_index,:,:],
            shading='gouraud',
            vmin=umin, vmax=umax)

    u_mag_quiv_pred_quiv = axs["u_mag_quiv_pred"].quiver(
                geom["grid2d_x"][::quiver_stepx,::quiver_stepy],geom["grid2d_y"][::quiver_stepx,::quiver_stepy], 
                data[f"u_pred"][timestep_index,:,:][::quiver_stepx,::quiver_stepy],
                data[f"v_pred"][timestep_index,:,:][::quiver_stepx,::quiver_stepy])

    axs["u_mag_quiv_pred"].axhline(y = y_value, color = 'r', linestyle = "solid")

    plt.colorbar(u_mag_quiv_pred_pcol, ax=axs["u_mag_quiv_pred"])

    #Show index used for extracting profile

    axs["u_mag_quiv_pred"].set_xlabel("X spatial index")
    axs["u_mag_quiv_pred"].set_ylabel("Y spatial index")
    axs["u_mag_quiv_pred"].set_title("Predicted velocity magnitude with stream lines")
    axs["u_mag_quiv_pred"].text(0.05,0.05,
        f"Meshsize: {geom['numx']}x{geom['numy']}",
        bbox={'facecolor':'w', 'alpha':0.5})

    #Pressure with quiver plot
    p_quiv_pcol_pred = axs["p_quiv_pred"].pcolormesh(
            geom["grid2d_x"],geom["grid2d_y"],
            data[f"p_pred"][timestep_index,:,:],
            shading='gouraud',
            vmin=pmin, vmax=pmax)

    p_quiv_quiv_pred = axs["p_quiv_pred"].quiver(
                geom["grid2d_x"][::quiver_stepx,::quiver_stepy],geom["grid2d_y"][::quiver_stepx,::quiver_stepy], 
                data[f"u_pred"][timestep_index,:,:][::quiver_stepx,::quiver_stepy],
                data[f"v_pred"][timestep_index,:,:][::quiver_stepx,::quiver_stepy])

    plt.colorbar(p_quiv_pcol_pred, ax=axs["p_quiv_pred"])

    axs["p_quiv_pred"].axhline(y = y_value, color = 'r', linestyle = "solid")
    axs["p_quiv_pred"].set_xlabel("X spatial index")
    axs["p_quiv_pred"].set_title("Predicted pressure with velocity stream lines")

    #U at x=0.8
    axs["u"].plot(data["u"][timestep_index,y_index,:],linestyle="solid", linewidth=1.5, color='black', label='U')
    axs["u"].plot(data["u_pred"][timestep_index,y_index,:],linestyle="dashed", linewidth=1.5, color='black', label='U pred')
    axs["u"].legend(fontsize="small", loc="upper right")
    axs["u"].set_ylabel("Velocity (u) m/s")
    axs["u"].set_title(f"Profiles extracted from red-line Y={y_value:0.1f}")

    axs["v"].plot(data["v"][timestep_index,y_index,:],linestyle="solid", linewidth=1.5, color='black', label='V')
    axs["v"].plot(data["v_pred"][timestep_index,y_index,:],linestyle="dashed", linewidth=1.5, color='black', label='V pred')
    axs["v"].legend(fontsize="small", loc="upper right")
    axs["v"].set_ylabel("Velocity (v) m/s")

    axs["p"].plot(data["p"][timestep_index,y_index,:],linestyle="solid", linewidth=1.5, color='black', label='P')
    axs["p"].plot(data["p_pred"][timestep_index,y_index,:],linestyle="dashed", linewidth=1.5, color='black', label='P pred')
    axs["p"].legend(fontsize="small", loc="upper right")
    axs["p"].set_xlabel("Y index-value")
    axs["p"].set_ylabel("Pressure (Pa)")

    output_path = (f"./plots/static.png")
    plt.savefig(output_path)
    
    return 0
