#!/bin/bash

# Path alle librerie acados
export LD_LIBRARY_PATH=/root/robotic-mpc/acados/lib:$LD_LIBRARY_PATH

# Path alla cartella sorgente acados (serve ad acados_ocp_solver)
export ACADOS_SOURCE_DIR=/root/robotic-mpc/acados

# export DYLD_LIBRARY_PATH=/Users/bb/Desktop/robotic-mpc/acados/lib:${DYLD_LIBRARY_PATH}
# export DYLD_FALLBACK_LIBRARY_PATH=/Users/bb/Desktop/robotic-mpc/acados/lib:${DYLD_FALLBACK_LIBRARY_PATH}
# export ACADOS_SOURCE_DIR=/Users/bb/Desktop/robotic-mpc/acados



python simulator.py
  