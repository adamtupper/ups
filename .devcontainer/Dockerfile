FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04

ENV PATH="${PATH}:/home/docker/.local/bin"

# Set timezone
ENV TZ=Canada/Eastern
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

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

# Install Git and Pip
RUN sudo apt-get install -y software-properties-common git python3-pip

# Install virtualenv
RUN pip3 install virtualenv

# Set the working directory
WORKDIR /home/docker/workspace/