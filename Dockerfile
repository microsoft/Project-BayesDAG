# This Dockerfile is used in the AML experiment runs (through the evaluation pipeline)
FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.6-cudnn8-ubuntu20.04

USER root:root

RUN apt-get update && \
    apt-get install -y \
    unixodbc unixodbc-dev && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*