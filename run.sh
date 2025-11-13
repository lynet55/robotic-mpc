#!/bin/bash
# Script to run the MPC with proper library paths

export LD_LIBRARY_PATH=/home/bb/Desktop/projects/robotic-mpc/acados/lib:$LD_LIBRARY_PATH
export ACADOS_SOURCE_DIR=/home/bb/Desktop/projects/robotic-mpc/acados

python main.py

