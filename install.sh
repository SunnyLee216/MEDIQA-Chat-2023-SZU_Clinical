#!/bin/bash

# Define environment name
env_name="SZU_Clinical_taskB_venv"

# Create conda environment
conda create --name $env_name

# Activate conda environment
conda activate $env_name

# Install requirements
conda install -y -c conda-forge --file requirements.txt

# Deactivate conda environment
conda deactivate