# README

## Project Motivation
Many computational simulations are too intensive to be useful, even with the use of modern supercomputing facilities. 

This is particularly true of fluid dynamics codes, and even more so, when the Reynolds number is very high.

Machine learning has scope to improve the performance of these simulations, by shifting much of the computation "offline" into the model training phase, leaving almost real-time simulation via inference.

### Project hypothesis
The performance of physical simulations using PDE's typically suffer when numerical stability constraints are applied such as the Courant-Freidrich-Lewy (CFL) condition, also known as the Courant number. 

## Courant number
Typically CFD codes will maintain a delta_t (time step value) that satisfies the CFD criteria.
The pseudo-code for this loop is:
- Compute initial velocity field
- Loop over cells in mesh and using cell-size and velocity compute cells CFL number
- Adjust time step so that the highest CFL number in the simulation (considering all cells) is below a tolerance value. (In LES this number is typically 0.5 to 1)

The "ML kernel" approach does not require a CFL condition as the inference is mesh-less.
Therefore the we could potentially "ignore" the smallest cells from the CFL field and therefore time step calculation.

If velocity is held constant smaller cells have a higher CFL number, therefore excluding them from the CFL calculation and time-step evaluation is likely to result in larger time-steps and therefore overall greater simulation performance.
  
## Hybrid ML-Simulation code
In the case of fluid simulations this is seen at the boundary conditions where low mean velocity requires very small cell sizes and timesteps.

This project hypothesizes that by replacing traditional simulation code in the regions of the mesh with very small cell size with machine-learning model kernels the overall performance of the code can be improved, whilst retaining the trusted solution of the main bulk of the simulation.

## Project plan

The full goal of this repository is to gradually build up knowledge of PINNs by using progressively more complex PDEs.

The first step is to solve the 1D heat equation explicitly to provide training data and to define initial and boundary conditions.

From this, a traditional deep neural-network will be constructed with which we will contrast a PINN will incorporates the physics conditions described in the explicit solver.

This process should showcase simply why a PINN is a superior choice for modelling fluid dynamics behavior.

From here we can increase the dimensionality of the PINN by extending it to the 2D heat equation, this will allow us to then change focus from the heat equation to more complex fluid PDEs such as the stream function.

Once a successful 2D PINN for a fluid PDE such as the stream function is produced we can take simulated LES data from OpenFOAM and the governing equation (this will need significant expertise in understanding the PDE and the initial/boundary conditions) to create an OpenFOAM-LES PINN.

The PINN process will require some degree of hyper-parameter tuning; particularly model architecture such as number of layers, weights, training epochs and activation functions.

The last step will be to edit OpenFOAM such that a hybrid PINN-solver can be implemented where the smallest cells near the boundary conditions are updated using the PINN but the central turbulent fluid is updated using the solver.

We will then benchmark the accuracy and performance of:

- OpenFOAM alone
- PINN alone
- Hybrid OpenFOAM-PINN solver

### Likely future work
PINNs are set in terms of their computational domain so if all the above works then using DeepONets would be desirable to generalise the problem.

Other model architecture incorporated into the DNN.

Use of accelerated libraries such as ZenDNN.