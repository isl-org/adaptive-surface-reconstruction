FROM ubuntu:24.04@sha256:a4453623f2f8319cfff65c43da9be80fe83b1a7ce689579b475867d69495b782

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
  && rm -rf /var/lib/apt/lists/*

# Miniforge
ENV PATH="/root/miniforge3/bin:${PATH}"
RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
    && bash Miniforge3-$(uname)-$(uname -m).sh -b \
    && rm -v Miniforge3*.sh \
    && conda --version

COPY docker_entrypoint.sh /docker_entrypoint.sh

ENTRYPOINT ["/docker_entrypoint.sh"]
