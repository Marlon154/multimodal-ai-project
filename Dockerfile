# Use an official CUDA-enabled Python runtime as a parent image
# FROM nvidia/cuda:11.4.3-devel-ubuntu20.04
FROM python:3.7.4-slim-buster

# Set environment variables
ENV PYTHONUNBUFFERED=1
#ENV CUDA_HOME=/usr/local/cuda-10.2

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    python3-pip \
    git \
    wget \
    libfreetype6-dev \
    libpng-dev \
    zlib1g-dev

# Create and activate conda environment
RUN pip install spacy==2.1.9
RUN pip install allennlp==0.9.0
RUN pip install scikit-learn==0.24.2
RUN pip install torch==1.5.1
RUN pip install torchvision==0.6.1
RUN pip install transformers==2.5.1
RUN pip install overrides==3.1.0
RUN pip install tensorboard==2.5.0
RUN pip install nltk==3.4.5
RUN pip install pymongo==3.10.1
RUN pip install pycocoevalcap==1.2.0
RUN pip install ptvsd
RUN pip install pudb
RUN pip install docopt==0.6.2
RUN pip install schema==0.7.7
RUN pip install textstat==0.7.3
RUN pip install cython==0.29.14

RUN python -m spacy download en_core_web_lg
RUN pip install nltk pymongo
RUN python -m nltk.downloader punkt
# Install the project
WORKDIR /app
COPY . /app
RUN python setup.py develop

CMD python tell/training/train.py
