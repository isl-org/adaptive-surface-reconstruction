FROM ubuntu:22.04@sha256:34577a83cd7a8f1e55f6eb173fd9bb70c16127a83d6bf06f56f6b1fff9e406b9

# Miniconda requires bash as the default shell.
SHELL ["/bin/bash", "-c"]


RUN apt-get update && apt-get install -y \
  build-essential \
  file \
  git \
  libglu1-mesa-dev \
  libtbb-dev \
  patchelf \
  wget \
  xorg-dev \
  wayland-protocols \
  libwayland-dev \
  libxkbcommon-dev \
  && rm -rf /var/lib/apt/lists/*

# Miniforge
ENV PATH="/root/miniforge3/bin:${PATH}"
RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
    && bash Miniforge3-$(uname)-$(uname -m).sh -b \
    && rm -v Miniforge3*.sh \
    && conda --version

COPY docker_entrypoint.sh /docker_entrypoint.sh

ENTRYPOINT ["/docker_entrypoint.sh"]
