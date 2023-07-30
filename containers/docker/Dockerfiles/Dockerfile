ARG OS_VER=latest

FROM ubuntu:${OS_VER} AS distro

# Version-independent base layer
FROM distro AS base0
SHELL ["/bin/bash", "-c"] 

ARG DEBIAN_FRONTEND=noninteractive

LABEL "author"="Harry McHugh"

RUN apt-get update \
 && apt-get -y install --no-install-recommends \
    apt-utils vim-tiny nano-tiny wget ca-certificates rsync ffmpeg \
    sudo passwd libnss-wrapper \
 && rm -rf /var/lib/apt/lists/*

# Version-specific runtime layer
FROM base0 AS runtime
ARG FOAM_VERSION=2212
ARG PACKAGE=openfoam${FOAM_VERSION}
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && wget -q -O - https://dl.openfoam.com/add-debian-repo.sh | bash \
 && apt-get update \
 && apt-get -y install --no-install-recommends ${PACKAGE} \
 && apt-get -y install python3 python3-pip git \
 && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
 && rm -rf /var/lib/apt/lists/*

COPY repo-key /

RUN chmod 600 /repo-key \  
  && echo "IdentityFile /repo-key" >> /etc/ssh/ssh_config \  
  && echo -e "StrictHostKeyChecking no" >> /etc/ssh/ssh_config \
  && git clone --depth 1 git@github.com:harrymchugh/pinns.git \ 
  && cd pinns \
  && pip install --upgrade pip \
  && pip install setuptools --upgrade \
  && pip install .
