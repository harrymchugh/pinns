# CFDPINN container

A docker container for reproducing the work in this project has been hosed in Docker Hub at the repository /harrymchugh/cfdpinn.

The container can be downloaded with docker pull for use on local workstations, public cloud VMs and anywhere else the Docker runtime is installed and active.

```
docker pull harrymchugh/cfdpinn
```

With the container availabe on your system you can now run scripts to reproduce the OpenFOAM simulations and nerual network training.



## Apptainer

Running docker on multi-tenant environments is usually problematic.

Apptainer is a similar container based solution better suited to multi-tenant systems such as ARCHER2 and Cirrus.

Apptainer (or singularity as it was previously known) can be installed via Spack if it is not already available via system wide install. 

Once apptainer is installed you can simply pull the cfdpinn docker container from Docker Hub using apptainer.

During this process apptainer will conver the Docker image into an apptainer SIF file format.

```
apptainer pull docker://harrymchugh/cfdpinn:latest
```

With the SIF file you can now execute OpenFOAM and cfdpinn training codes on whatever hardware you have available.

To run the cavity OpenFOAM simulation:

```
apptainer run --no-home -B $CFDPINN_ROOT/openfoam/scripts:/scripts -B $CFDPINN_ROOT/openfoam/cases:/cases cfdpinn.sif /bin/bash -c /scripts/cavity.sh
```

To run the CFDPINN training process on the cavity simulation data

```
apptainer run --no-home -B $CFDPINN_ROOT/data/cavity:/cavity -B $CFDPINN_ROOT/training:/training cfdpinn.sif /bin/bash -c /training/cfdpinn.sh
```

## Using NVIDIA GPUs

Neural network training time can be sigificantly sped up by using GPUs.

Docker can expose GPUs attached to the host system by adding additional arguments to docker run.

Please note that the nvidia-container-toolkit must be installed on the host system.

```
docker run --gpus all harrymchugh/cfdpinn 
```

Apptainer can expose GPUs attached to the host system by running the `--nv` flag to the apptainer run command.

Apptainer acheives this be searching the host system for the NVIDIA driver and CUDA runtime libraries and bindmounting them into the container.
Should the drivers or libraries be missing or installed in uncommon locations on the host, this may not work.

```
apptainer run --nv --no-home -B $CFDPINN_ROOT/data/cavity:/cavity -B $CFDPINN_ROOT/training:/training cfdpinn.sif /bin/bash -c /training/cfdpinn.sh

```

The docker approach has been tested in a Google Cloud VM with an L4 GPU attached:

- Training time using CPU:
- Training time using GPU:

The apptainer approach has been tested on the Cirrus GPU nodes using a single V100 card.

- Training time using CPU:
- Training time using GPU:

Please note that the use of AMD GPUs has not been tested.

## Building CFDPINN.sif manually

The reccommended way to otain the cfdpinn container image is to use Docker/Apptainer pull to download the container in the appropriate format from Docker Hub.

However, should you wish to build the container manually Dockerfile and apptainer definition files are provided in the apptainer/defs and Docker/Dockerfiles directories.
