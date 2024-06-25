# FROM python:3.7.4-slim-buster
FROM nvidia/cuda:11.4.3-devel-ubuntu20.04

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    python3-pip \
    git \
    wget \
    libfreetype6-dev \
    libpng-dev \
    zlib1g-dev 

# install python 3.7.4


RUN export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

RUN pip install packaging
RUN pip install torch==1.5.1

RUN git clone https://github.com/NVIDIA/apex
WORKDIR ./apex
RUN pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./

RUN pip install spacy==2.1.9
RUN pip install allennlp==0.9.0
RUN pip install scikit-learn==0.24.2
RUN pip install torchvision==0.6.1
RUN pip install transformers==2.5.1
RUN pip install overrides==3.1.0
RUN pip install tensorboard==2.5.0
RUN pip install nltk==3.4.5
RUN pip install pymongo==3.10.1
RUN pip install pycocoevalcap==1.2.0

RUN python3 -m spacy download en_core_web_lg
RUN pip install nltk pymongo
RUN python3 -m nltk.downloader punkt

RUN pip install ptvsd
RUN pip install pudb
RUN pip install docopt==0.6.2
RUN pip install schema==0.7.7
RUN pip install textstat==0.7.3


WORKDIR /app
COPY . /app
RUN python3 setup.py develop

RUN ln -s /usr/bin/python3 /usr/bin/python

CMD tell train expt/nytimes/9_transformer_objects/config.yaml -f