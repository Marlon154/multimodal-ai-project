# Use an official CUDA-enabled Python runtime as a parent image
# FROM nvidia/cuda:12.4.0-devel-ubuntu22.04
FROM nvidia/cuda:11.1.1-devel-ubuntu20.04
# from ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    bzip2 \
    ca-certificates \
    git \
    sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install conda (ubuntus package manager is awesome - this would've taken me 10 seconds on arch linux)
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && /bin/bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH
RUN conda update -n base -c defaults conda

# init conda for bash
RUN conda init bash

# install requirements | setup environment
WORKDIR app
COPY environment.yml .
RUN conda env create -f environment.yml
RUN echo "conda activate tell" >> ~/.bashrc

# CUDA home
# ENV CUDA_HOME=/usr/local/cuda-11.1

# install apex
# WORKDIR /home/docker
# RUN git clone https://github.com/NVIDIA/apex.git
# WORKDIR /home/docker/apex
# RUN git checkout 44532b3
# RUN /bin/bash -c 'source /opt/conda/bin/activate && conda activate tell && pip install -v --no-cache-dir --global-option="--pyprof" --global-option="--cpp_ext" --global-option="--cuda_ext" ./'
# WORKDIR /home/docker/app

# install tell
COPY . .
RUN /bin/bash -c 'source /opt/conda/bin/activate && conda activate tell && python setup.py develop'

# download python package stuff
RUN /bin/bash -c 'source /opt/conda/bin/activate && conda activate tell && python -m spacy download en_core_web_lg && python -m nltk.downloader punkt'

# train model
CMD /bin/bash -c 'source /opt/conda/bin/activate && conda activate tell && CUDA_VISIBLE_DEVICES=0 tell train expt/nytimes/9_transformer_objects/config.yaml -f'
# CMD bash
