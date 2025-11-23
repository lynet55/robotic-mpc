#!/bin/bash
# Script to run the MPC with proper library paths

export DYLD_LIBRARY_PATH=/Users/bb/Desktop/robotic-mpc/acados/lib:${DYLD_LIBRARY_PATH}
export DYLD_FALLBACK_LIBRARY_PATH=/Users/bb/Desktop/robotic-mpc/acados/lib:${DYLD_FALLBACK_LIBRARY_PATH}
export ACADOS_SOURCE_DIR=/Users/bb/Desktop/robotic-mpc/acados

python main.py

