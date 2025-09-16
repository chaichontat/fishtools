FROM mambaorg/micromamba:cuda12.8.1-ubuntu24.04

# Ensure subsequent RUN lines can use bash and we can run inside envs succinctly
SHELL ["/bin/bash", "-lc"]

# Copy and resolve the Conda environment first for better build caching
ARG MAMBA_USER=mambauser
WORKDIR /home/${MAMBA_USER}/app

# Install OS packages needed during build/runtime
USER root
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
USER ${MAMBA_USER}

COPY --chown=${MAMBA_USER}:${MAMBA_USER} environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && micromamba clean --all --yes

COPY --chown=${MAMBA_USER}:${MAMBA_USER} environment_rsc_rapids_25.04.yml /tmp/environment_rsc_rapids_25.04.yml
RUN micromamba install -y -n base -f /tmp/environment_rsc_rapids_25.04.yml && micromamba clean --all --yes

COPY . .

# Install the package into the created environment
RUN micromamba run pip install --no-cache-dir -e .
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Default command: drop into the environment's shell
CMD ["python", "-c", "import fishtools; print('fishtools import OK')"]
