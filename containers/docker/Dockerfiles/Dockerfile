ARG OS_VER=latest

FROM ubuntu:${OS_VER} AS distro

# Version-independent base layer
FROM distro AS base0
ARG DEBIAN_FRONTEND=noninteractive

LABEL "author"="Harry McHugh"

RUN apt-get update \
 && apt-get -y install --no-install-recommends \
    apt-utils vim-tiny nano-tiny wget ca-certificates rsync \
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
 && apt-get -y install python3 python3-pip \
 && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
 && rm -rf /var/lib/apt/lists/*

ENTRYPOINT [ ". /usr/lib/openfoam/openfoam2212/etc/bashrc" ]
