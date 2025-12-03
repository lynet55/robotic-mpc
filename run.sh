#!/bin/bash
# Script to run the MPC with proper library paths

export LD_LIBRARY_PATH=/home/bb/Desktop/projects/robotic-mpc/acados/lib:$LD_LIBRARY_PATH
export ACADOS_SOURCE_DIR=/home/bb/Desktop/projects/robotic-mpc/acados

#For Mac
export DYLD_LIBRARY_PATH=/Users/bb/Desktop/robotic-mpc/acados/lib:${DYLD_LIBRARY_PATH}
export DYLD_FALLBACK_LIBRARY_PATH=/Users/bb/Desktop/robotic-mpc/acados/lib:${DYLD_FALLBACK_LIBRARY_PATH}
export ACADOS_SOURCE_DIR=/Users/bb/Desktop/robotic-mpc/acados

python main.py