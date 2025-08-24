# Base image matching the agent's environment
FROM ubuntu:24.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install all necessary build dependencies and tools
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    ninja-build \
    clang \
    python3 \
    libssl-dev \
    pkg-config \
    ssh-client \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory for the project
WORKDIR /work