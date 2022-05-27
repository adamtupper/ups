FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04

ENV PATH="${PATH}:/home/docker/.local/bin"

RUN apt-get update && \
    apt-get install -y sudo

# Add (and switch to) new 'docker' user
RUN groupadd -g 110 usergroup
RUN adduser --disabled-password --gecos '' docker
RUN adduser docker sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER docker

# Create workspace directory
RUN mkdir -p /home/docker/workspace/

# Install Git, Python, and Pip
RUN sudo apt-get update && \
    sudo apt-get install -y software-properties-common git && \ 
    sudo add-apt-repository ppa:deadsnakes/ppa && \
    sudo apt-get install -y  python3.9

# Install virtualenv
RUN pip install virtualenv

# Set the working directory
WORKDIR /home/docker/workspace/