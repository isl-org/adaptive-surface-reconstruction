FROM ubuntu:22.04

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

# Miniconda
ENV PATH="/root/miniconda3/bin:${PATH}"
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && bash Miniconda3-latest-Linux-x86_64.sh -b \
 && rm Miniconda3-latest-Linux-x86_64.sh \
 && conda --version


COPY docker_entrypoint.sh /docker_entrypoint.sh

ENTRYPOINT ["/docker_entrypoint.sh"]
