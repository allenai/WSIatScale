FROM python:3.7-stretch

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PATH /usr/local/nvidia/bin/:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
LABEL com.nvidia.volumes.needed="nvidia_driver"

WORKDIR /stage

# Install base packages.
RUN apt-get update --fix-missing && apt-get install -y \
    bzip2 \
    ca-certificates \
    curl \
    gcc \
    git \
    libc-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    libevent-dev \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch
RUN pip install --no-cache-dir transformers -q
Run pip install --no-cache-dir tqdm

Run git clone https://github.com/NVIDIA/apex
WORKDIR /stage/apex
Run git checkout 0ac5dd6253d3954a4875b5648a9af439b5ae7b4e
Run pip3 install -v --no-cache-dir ./

WORKDIR /stage/
COPY . .


ENV PYTHONPATH=.

CMD ["/bin/bash"]