# CFDPINN container

A docker container for reproducing the work in this project has been hosed in Docker Hub at the repository /harrymchugh/cfdpinn.

The container can be downloaded with docker pull for use on local workstations, public cloud VMs and anywhere else the Docker runtime is installed and active.

```
docker pull harrymchugh/cfdpinn
```

With the container available on your system you can now run scripts to reproduce the OpenFOAM simulations and neural network training.

## Apptainer

Running docker containers on multi-tenant environments is usually problematic.

Apptainer is a similar container based solution better suited to multi-tenant systems such as ARCHER2 and Cirrus.

Apptainer (or singularity as it was previously known) can be installed via Spack if it is not already available via system wide install. 

Once Apptainer is installed you can simply pull the CFDPINN docker container from Docker Hub using Apptainer.

During this process Apptainer will convert the Docker image into an Apptainer SIF file format.

```
apptainer pull docker://harrymchugh/cfdpinn:latest
```

With the SIF file you can now execute OpenFOAM and CFDPINN training codes on whatever hardware you have available.

### TMPDIR and running out of space error
For large containers such as the CFDPINN container a common issue that users can run into is an error that suggests there is a lack of space. This is due to the way Apptainer uses temporary directories when building/pulling containers.

It can typically be rectified by setting the APPTAINER_TMPDIR directory to somewhere with lots of storage available.

```
mkdir apptainer_tmp                  
export APPTAINER_TMPDIR=$PWD/apptainer_tmp
```

Once the SIF file is successfully created the apptainer_tmp directory can safely be deleted.



## OpenFOAM and PINN training

Clone this repository and set the $CFDPINN_ROOT environment variable to the root of the repository.

```
git clone --depth 1 git@github.com:harrymchugh/pinns.git

cd pinns
export CFDPINN_ROOT=$PWD

mkdir -p $CFDPINN_ROOT/profiles/tboard
mkdir -p $CFDPINN_ROOT/tboard
mkdir -p $CFDPINN_ROOT/models
mkdir -p $CFDPINN_ROOT/plots
```

To run the cavity OpenFOAM simulation with Docker or Apptainer:

```
apptainer run -B $CFDPINN_ROOT:/mnt cfdpinn.sif /bin/bash -c /mnt/openfoam/scripts/cavity-nu0.01-U1-20x20.sh
```

```
docker run -v $CFDPINN_ROOT:/mnt harrymchugh/cfdpinn /bin/bash -c /mnt/openfoam/scripts/cavity-nu0.01-U1-20x20.sh
```

Once the OpenFOAM simulations are complete you can run the CFDPINN network training process on the cavity simulation data.

```
apptainer run -B $CFDPINN_ROOT:/mnt cfdpinn.sif /bin/bash -c /mnt/training/training.sh
```

```
docker run -v $CFDPINN_ROOT:/mnt harrymchugh/cfdpinn /bin/bash -c /mnt/training/training.sh
```

Similarly the inference scripts can be run in the same manner:

```
apptainer run -B $CFDPINN_ROOT:/mnt cfdpinn.sif /bin/bash -c /mnt/inference/inference.sh
```

```
docker run -v $CFDPINN_ROOT:/mnt harrymchugh/cfdpinn /bin/bash -c /mnt/inference/inference.sh
```

## Using NVIDIA GPUs

Neural network training time, and inference, can be significantly sped up by using GPUs.

Docker can expose GPUs attached to the host system by adding additional arguments to docker run.

Please note that the NVIDIA-container-toolkit must be installed on the host system.

```
docker run --gpus all <docker args> 
```

Apptainer can expose GPUs attached to the host system by running the `--nv` flag to the apptainer run command.

Apptainer achieves this be searching the host system for the NVIDIA driver and CUDA runtime libraries and bind mounting them into the container.

Should the drivers or libraries be missing or installed in uncommon locations on the host, this may not work.

```
apptainer run --nv <apptainer args>
```

### Testing
The docker approach has been tested in a Google Cloud VM with an L4 GPU attached:

The apptainer approach has been tested on the Cirrus GPU nodes using a single V100 card.

Please note that the use of AMD GPUs has not been tested.

## Building CFDPINN container locally

The recommended way to obtain the CFDPINN container image is to use Docker/Apptainer to pull/download the container in the appropriate format from Docker Hub.

However, should you wish to build the container manually a Dockerfile file is provided [here](./docker/Dockerfiles/Dockerfile). Change directory to the location of the Dockerfile and execute the following commands.

```
sudo docker build --no-cache --tag harrymchugh/cfdpinn .
```

To build the Docker image you must have sudo access.

## Tensorboard
Tensorboard is an excellent tool for monitoring, debugging and analysing performance of PyTorch.

To start a Tensorboard server and expose its data for viewing in a web browser local to the host OS running the container run the following commands.

```
docker run -p 8888:8888 -p 6006:6006 -v $CFDPINN_ROOT:/mnt harrymchugh/cfdpinn tensorboard --logdir=/mnt/tboard --port 6006 --bind_all
```

and for Apptainer:

```
apptainer run -B $CFDPINN_ROOT:/mnt cfdpinn.sif tensorboard --logdir=/mnt/tboard --port 6006 --bind_all
```