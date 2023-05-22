FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04

# useful tools
RUN apt-get update \
 && apt-get install -y \
        build-essential \
        cmake \
        cppcheck \
        gdb \
        git \
        sudo \
        vim \
        wget \
        tmux \
        curl \
        less \
        htop \
        libsm6 libxext6 libgl1-mesa-glx libxrender-dev \
 && apt-get clean

# Add a user with the same user_id as the user outside the container
# Requires a docker build argume--------------------------------+++++++++++++++++++++nt `user_id`
ARG USER=yuhengq
ARG GROUP=docker
ARG UID=1055
ARG GID=403

ENV USERNAME yuhengq

RUN apt-get update -y && apt-get -y install sudo && sudo apt-get -y update && sudo apt-get -y upgrade

RUN sudo groupadd -g ${GID} ${GROUP} && \
    sudo useradd -g ${GID} -u ${UID} -m -s /bin/bash ${USER} && \
    id ${USER} && \
    usermod -aG root ${USER} && \
    mkdir /app && sudo chown -R ${USER} /app && \
    adduser ${USER} sudo && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER $USER

 # install miniconda
ENV MINICONDA_VERSION latest
ENV CONDA_DIR /home/${USER}/miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -O ~/miniconda.sh  && \
    chmod +x ~/miniconda.sh
RUN bash ~/miniconda.sh -fbp $CONDA_DIR && \
    rm ~/miniconda.sh
# make non-activate conda commands available
ENV PATH="${CONDA_DIR}/bin:$PATH"

# make conda activate command available from /bin/bash --interative shells
RUN conda update -n base -c defaults conda

ARG CONDA_ENV=pypose

RUN conda create -n ${CONDA_ENV} python=3.8
# RUN conda install -n ${CONDA_ENV} -c pytorch -c conda-forge scipy matplotlib tensorboard debugpy tqdm scipy opencv=4.5
RUN ${CONDA_DIR}/envs/${CONDA_ENV}/bin/pip install pypose

RUN conda init bash
WORKDIR /home/${USER}