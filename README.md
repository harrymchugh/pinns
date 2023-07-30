# CFDPINN

**C**omputational **F**luid **D**ynamics **P**hysics **I**nformed **N**eural **N**etwork

**Author**: Harry McHugh \
**Supervisor**: Adrian Jackson

**Dissertation Title**: \
```On the suitability of physics informed neural networks to accelerate computational fluid dynamics ```

![Static plot showing CFDPINN output](./images/static.png)
## Project motivation
Many computational simulations are too intensive to be useful, even with the use of modern supercomputing facilities. 

This is particularly true of fluid dynamics codes, and even more so, when the Reynolds number is very high.

Machine learning has scope to improve the performance of these simulations, by shifting much of the computation "offline" into the model training phase, leaving almost real-time simulation via inference.

## Project aims
Physics informed neural networks have been seen to perform better than traditional deep learning models when learning physical laws.

Therefore, this project aims to acheive the following goals:

- Produce a PINN that is capable of accurately predicting 2D fluid-flow properties
- Compare the performance and process of training PINNs with traditional methods.
- Test multiple scenarios to ascertain under what conditions PINNs succeed or fail. 

By achieving these aims we feel we will be able to provide a meaningful commentary on the question; “To what extent PINNs are able to accelerate CFD?”

## Installing CFDPINN
### Containers
The recommended method for reproducing the work described in the accompanying dissertation is to obtain the Docker image with all binaries pre-built and installed inside the container.

For detailed instructions on how to obtain the docker container please consult the following [README](./containers/README.md).

### Manual installation
Should you wish to install the source code for CFDPINN yourself you can clone this repository and simply install the CFDPINN application by running the following command in the root of the directory.

```
git clone --depth 1 git@github.com:harrymchugh/pinns.git
cd pinns
pip install -e .
```

## Running CFDPINN
There are three main stages to reproducing the work in this dissertation.

The first is to generate simulation data which acts as a reference to current industry methods for generating fluid properties with the computational fluid dynamics application OpenFOAM.

Once the OpenFOAM simulation is complete we can use the CFDPINN application in this repository to train a physics-informed neural network (PINN).

Additionally pre-trained models can be used to predict fluid properties in *"inference-only"* mode varying certain parameters such as output mesh and geometry.

Example scripts are provided for OpenFOAM [simulations](./openfoam/scripts/cavity-nu0.01-U1-20x20.sh), [PINN training](./training/training.sh) and [inference](./inference/inference.sh).

### Running CFDPINN manually
These scripts assume they will be run inside the supplied container. Detailed instructions on how to run each stage using the supplied containers are provided [here](./containers/README.md).

Should you wish to run these locally please ensure your environment is setup so that the applications and libraries are accessible and the paths used in the provided scripts updated to point to your specific filesystem locations.

## CFDPINN application arguments
The CFDPINN application has been written to ingest many parameters at the command line to provide flexibility in its use cases.

For full observation of the available parameters in CFDPINN run the following command.

```
cfdpinn -h

CFD PINN

options:
  -h, --help            show this help message and exit
  --debug               Enable debug output
  --case-type {cavity}  CFDPINN currently only supports OpenFOAM simulations for lid-driven cavity flow
  --case-dir CASE_DIR   The directory of the OpenFOAM case used for PINN training
  --start-time START_TIME
                        The first value of the OpenFOAM output timesteps
  --end-time END_TIME   The last value of the OpenFOAM output timesteps
  --sim-dt SIM_DT       The original timestep value of the OpenFOAM case
  --load-dt LOAD_DT     The desired timestep for the loaded OpenFOAM case. This allows the user to decimate the OpenFOAM simulation
                        data. For example a simulation run at 0.01 can be loaded at 0.05 to reduce training time
  --startx STARTX       The x value of the first cell
  --starty STARTY       The y value of the first cell
  --endx ENDX           The x value of the last cell
  --endy ENDY           The y value of the last cell
  --numx NUMX           The number of cells in the x-dimension of the OpenFOAM case
  --numy NUMY           The number of cells in the y-dimension of the OpenFOAM case
  --viscosity VISCOSITY
                        The viscosity of the fluid used in the OpenFOAM case
  --initial_u INITIAL_U_LID
                        The u (x) velocity of the fluid flowing over the cavity top wall
  --num-frames NUM_FRAMES
                        Number of frames for the fluid animations
  --training-animation  Output MP4 showing training locations with training U_mag,U,V,P over all timesteps
  --prediction-animation
                        Output MP4 showing training locations with predicted U_mag,U,V,P over all timesteps
  --residual-animation  Output MP4 showing training locations with residual U_mag,U,V,P over all timesteps
  --animations-path ANIMATIONS_PATH
                        The output path for the animations
  --static-plots        Produce static plots for analysis
  --static-plots-path STATIC_PLOTS_PATH
                        The output path for the static plots
  --load-simulation     Load simulation data regardless of training or inference
  --no-train            Don't train a model, only inference
  --test-percent TEST_SIZE
                        The percentage of OpenFOAM case data to retain to use as in the testing function of the PINN training
                        process
  --save-scaler-path SAVE_SCALER_PATH
                        The output path for the data scalar generated alongside a given input dataset. Required for future use of
                        the model on new data
  --load-scaler-path LOAD_SCALER_PATH
                        The path to load the data scalar for use with inference mode
  --device {cuda,cpu}   The type of device used for PyTorch operations, default is to use GPU is available but fall back to CPU if a
                        GPU cannot be found
  --save-model-path PINN_OUTPUT_PATH
                        Full path for the output of the trained PINN
  --optimizer {adam,sgd}
                        The type of optimizer used for model training
  --load-model-path LOAD_MODEL_PATH
                        Path to load a model in inference only mode
  --profile             Run profiling on train and inference
  --profile-path PROFILE_PATH
                        The path to save Tensorboard profiling outputs
  --tensorboard         Log training adn testing metrics with Tensorboard functionality
  --tensorboard-path TENSORBOARD_PATH
                        The path to save Tensorboard training/testing metrics
  --epochs EPOCHS       The number of epochs to train the PINN
  --lr LEARNING_RATE    The learning rate of the ADAM Optimizer used to train the PINN
  --inference-timing    Report timings for inference of fluid properties
  --adaption {lrannealing,softadapt,noadaption}
                        The type of loss function weighting to use
```
## Tensorboard; debugging, profiling and performance
In addition to simply running OpenFOAM simulation, training and inference CFDPINN allows users to use Tensorboard to interrogate profiling data (to assess PyTorch performance) and to view machine learning metrics such as testing and training losses in real-time or once training is complete.

Once again the recommended method for viewing Tensorboard and the accompanying data is to use the supplied Docker container which has Tensorboard built in. 

For instructions on how to start Tensorboard and expose it's server for viewing in a local web browser please see the "Tensorboard" section of the container [README](./containers/README.md).

Should you have Tensorboard installed locally you can simply start a Tensorboard session as follows.

```
tensorboard --logdir=/path/to/tboarddata
```

This will start a server and a link will be printed that can be navigated to in order to view the data.