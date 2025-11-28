#!/bin/bash

# Path alle librerie acados
export LD_LIBRARY_PATH=/mnt/c/Users/Crist/robotic-mpc/acados/lib:$LD_LIBRARY_PATH

# Path alla cartella sorgente acados (serve ad acados_ocp_solver)
export ACADOS_SOURCE_DIR=/mnt/c/Users/Crist/robotic-mpc/acados

python3 main.py


