FROM mambaorg/micromamba:cuda12.8.1-ubuntu24.04

# RUN apt-get update && \
#     apt-get install -y git && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/env.yaml
# RUN micromamba install -y -n base -f /tmp/env.yaml && \
#     micromamba clean --all --yes

# Copy package files
COPY . .
RUN micromamba env create -n fishtools -f ./environment.yml
RUN pip install --no-cache-dir -e .
