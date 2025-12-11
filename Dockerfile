FROM continuumio/miniconda3

# Install system dependencies for acados and Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    liblapack-dev \
    libblas-dev \
    libeigen3-dev \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pinocchio with visualize dependencies (meshcat required for MeshcatVisualizer)
RUN conda install -y -c conda-forge pinocchio meshcat-python && conda clean -afy

WORKDIR /root/robotic-mpc

# Clone and build acados (using HPIPM solver, qpOASES disabled due to GCC 14 incompatibility)
RUN git clone https://github.com/acados/acados.git && \
    cd acados && \
    git submodule update --recursive --init && \
    mkdir -p build && \
    cd build && \
    cmake -DACADOS_WITH_QPOASES=ON .. && \
    make install -j4

# Set acados environment variables
ENV LD_LIBRARY_PATH=/root/robotic-mpc/acados/lib:$LD_LIBRARY_PATH
ENV ACADOS_SOURCE_DIR=/root/robotic-mpc/acados

# Install acados Python interface
RUN pip install --no-cache-dir -e /root/robotic-mpc/acados/interfaces/acados_template

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Working directory for the source code
WORKDIR /root/robotic-mpc/src

CMD ["python", "simulator.py"]
