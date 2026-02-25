FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-venv \
    git curl ca-certificates wget unzip \
    build-essential cmake \
    libglib2.0-0 libgl1 libx11-6 libxext6 libxrender1 libsm6 libxi6 libxkbcommon0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

ARG SOFA_VERSION=23.06.00
ARG SOFA_URL="https://github.com/sofa-framework/sofa/releases/download/v${SOFA_VERSION}/SOFA_v${SOFA_VERSION}_Linux.zip"
RUN mkdir -p /opt && \
    curl -L "${SOFA_URL}" -o /tmp/sofa.zip && \
    unzip -q /tmp/sofa.zip -d /opt && \
    rm /tmp/sofa.zip

ENV SOFA_ROOT=/opt/SOFA_v${SOFA_VERSION}_Linux
ENV PYTHONPATH=
ENV PYTHONPATH=${SOFA_ROOT}/plugins/SofaPython3/lib/python3/site-packages:${PYTHONPATH}
ENV LD_LIBRARY_PATH=${SOFA_ROOT}/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

ARG STEVE_REF=f909e7122d846a3c2fec91eedddaa0ba88b6391d
ARG STEVE_RL_REF=64222b7b74d113a6e965b58d5739b475f2c87a02
ARG STEVE_BENCH_REF=1c7640eca7ed3701d1651cedbfad8f5b71132ff2
ARG STEVE_TRAINING_REF=f721647e4521e3c4c286dbbe832f26b3ab34bf60
ARG AORTIC_ARCH_REF=71a65cdac4eca9476344b590f6c7d97d1e8121db
ARG TORCH_VERSION=2.1.0
ARG TORCH_CUDA=cu118

WORKDIR /opt
RUN python3 -m pip install --no-cache-dir \
    torch==${TORCH_VERSION}+${TORCH_CUDA} \
    --index-url https://download.pytorch.org/whl/${TORCH_CUDA}

RUN git config --global url."https://github.com/".insteadOf "git@github.com:"

RUN git clone https://github.com/lkarstensen/stEVE.git eve && \
    cd eve && git checkout "${STEVE_REF}" && \
    python3 -m pip install -e .

RUN git clone https://github.com/lkarstensen/stEVE_bench.git eve_bench && \
    cd eve_bench && git checkout "${STEVE_BENCH_REF}" && \
    python3 -m pip install -e .

RUN git clone https://github.com/lkarstensen/stEVE_rl.git eve_rl && \
    cd eve_rl && git checkout "${STEVE_RL_REF}" && \
    python3 -m pip install -e .

RUN git clone https://github.com/lkarstensen/2023_IJCARS_aortic_arch_generator.git aorticarchgenerator && \
    cd aorticarchgenerator && git checkout "${AORTIC_ARCH_REF}" && \
    python3 -m pip install -e .

RUN git clone --recurse-submodules https://github.com/lkarstensen/stEVE_training.git eve_training && \
    cd eve_training && git checkout "${STEVE_TRAINING_REF}"

WORKDIR /workspace/steve_recommender
COPY . /workspace/steve_recommender
RUN python3 -m pip install -e .

RUN mkdir -p /workspace/results
ENV RESULTS_DIR=/workspace/results

CMD ["/bin/bash"]
