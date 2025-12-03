#!/bin/bash

# Path alle librerie acados
export LD_LIBRARY_PATH=/root/robotic-mpc/acados/lib:$LD_LIBRARY_PATH

# Path alla cartella sorgente acados (serve ad acados_ocp_solver)
export ACADOS_SOURCE_DIR=/root/robotic-mpc/acados



python simulator.py
