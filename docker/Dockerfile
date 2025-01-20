# Use the Miniconda3 image as the base
FROM continuumio/miniconda3:latest

# Set the working directory
WORKDIR /workspace

# Install missing system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libarchive-dev \
    && rm -rf /var/lib/apt/lists/*

# Install mamba into the base environment
RUN conda install -n base -c conda-forge -y mamba

# Use mamba to install the required Python libraries
RUN mamba install -n base -c conda-forge -y \
    scikit-learn \
    xgboost \
    lightgbm \
    catboost \
    && conda clean -afy

# Default command to run when starting the container
CMD ["bash"]

