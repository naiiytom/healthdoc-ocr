# Using official nvidia cuda 10.2 and cudnn7 on Ubuntu 18.04

ARG FROM_IMAGE=anibali/pytorch:1.5.0-cuda10.2
FROM ${FROM_IMAGE}

#Install python
RUN apt-get update && \
    apt-get install -y \
    gcc \
    build-essential

RUN pip3 install -r requirements.txt