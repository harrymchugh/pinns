# README

The full goal of this repository is to gradually build up knowledge of PINNs by using progressively more complex PDEs.

The first step is to solve the 1D heat equation explicitly to provide training data and to define initial and boundary conditions.

From this, a traditional deep neural-network will be constructed with which we will contrast a PINN will incorporates the physics conditions described in the explicit solver.

This process should showcase simply why a PINN is a superior choice for modelling fluid dynamics behavior.

From here we can increase the dimensionality of the PINN by extending it to the 2D heat equation, this will allow us to then change focus from the heat equation to more complex fluid PDEs such as the stream function.

Once a successful 2D PINN for a fluid PDE such as the stream function is produced we can take simulated LES data from OpenFOAM and the governing equation (this will need significant expertise in understanding the PDE and the initial/boundary conditions) to create an OpenFOAM-LES PINN.

The last step will be to edit OpenFOAM such that a hybrid PINN-solver can be implemented where the smallest cells near the boundary conditions are updated using the PINN but the central turbulent fluid is updated using the solver.

We will then benchmark the accuracy and performance of:

- OpenFOAM alone
- PINN alone
- Hybrid OpenFOAM-PINN solver