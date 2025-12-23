#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#





# define sources and use make's implicit rules to generate object files (*.o)

# model
MODEL_SRC=

MODEL_SRC+= six_dof_robot_aa12ba10_model/six_dof_robot_aa12ba10_dyn_disc_phi_fun.c
MODEL_SRC+= six_dof_robot_aa12ba10_model/six_dof_robot_aa12ba10_dyn_disc_phi_fun_jac.c
MODEL_SRC+= six_dof_robot_aa12ba10_model/six_dof_robot_aa12ba10_dyn_disc_phi_fun_jac_hess.c

MODEL_OBJ := $(MODEL_SRC:.c=.o)
# optimal control problem - mostly CasADi exports
OCP_SRC=


OCP_SRC+= six_dof_robot_aa12ba10_cost/six_dof_robot_aa12ba10_cost_y_0_fun.c
OCP_SRC+= six_dof_robot_aa12ba10_cost/six_dof_robot_aa12ba10_cost_y_0_fun_jac_ut_xt.c
OCP_SRC+= six_dof_robot_aa12ba10_cost/six_dof_robot_aa12ba10_cost_y_0_hess.c
OCP_SRC+= six_dof_robot_aa12ba10_cost/six_dof_robot_aa12ba10_cost_y_fun.c
OCP_SRC+= six_dof_robot_aa12ba10_cost/six_dof_robot_aa12ba10_cost_y_fun_jac_ut_xt.c
OCP_SRC+= six_dof_robot_aa12ba10_cost/six_dof_robot_aa12ba10_cost_y_hess.c

OCP_SRC+= acados_solver_six_dof_robot_aa12ba10.c

OCP_OBJ := $(OCP_SRC:.c=.o)
# for target example
EX_SRC= main_six_dof_robot_aa12ba10.c
EX_OBJ := $(EX_SRC:.c=.o)
EX_EXE := $(EX_SRC:.c=)


# combine model, (potentially) sim and ocp object files
OBJ=
OBJ+= $(MODEL_OBJ)
OBJ+= $(OCP_OBJ)

EXTERNAL_DIR=
EXTERNAL_LIB=

INCLUDE_PATH = /root/robotic-mpc/acados/include
LIB_PATH = /root/robotic-mpc/acados/lib

# preprocessor flags for make's implicit rules
CPPFLAGS+= -I$(INCLUDE_PATH)
CPPFLAGS+= -I$(INCLUDE_PATH)/acados
CPPFLAGS+= -I$(INCLUDE_PATH)/blasfeo/include
CPPFLAGS+= -I$(INCLUDE_PATH)/hpipm/include


# define the c-compiler flags for make's implicit rules
CFLAGS = -fPIC -std=c99   -O2

# # Debugging
# CFLAGS += -g3 -fno-diagnostics-show-line-numbers -g

# linker flags
LDFLAGS+= -L$(LIB_PATH)


# link to libraries
LDLIBS+= -lacados
LDLIBS+= -lhpipm
LDLIBS+= -lblasfeo
LDLIBS+= -lm
LDLIBS+= 

# libraries
LIBACADOS_SOLVER=libacados_solver_six_dof_robot_aa12ba10.so
LIBACADOS_OCP_SOLVER=libacados_ocp_solver_six_dof_robot_aa12ba10.so

# virtual targets
.PHONY : all clean
all: clean example
shared_lib: ocp_shared_lib

# some linker targets
example: $(EX_OBJ) $(OBJ)
	$(CC) $^ -o $(EX_EXE) $(LDFLAGS) $(LDLIBS)


bundled_shared_lib: $(OBJ)
	$(CC) -shared $^ -o $(LIBACADOS_SOLVER) $(LDFLAGS) $(LDLIBS)

ocp_shared_lib: $(OCP_OBJ) $(MODEL_OBJ)
	$(CC) -shared $^ -o $(LIBACADOS_OCP_SOLVER) $(LDFLAGS) $(LDLIBS) \
	-L$(EXTERNAL_DIR) -l$(EXTERNAL_LIB)


# Cython targets
ocp_cython_c: ocp_shared_lib
	cython \
	-o acados_ocp_solver_pyx.c \
	-I $(INCLUDE_PATH)/../interfaces/acados_template/acados_template \
	$(INCLUDE_PATH)/../interfaces/acados_template/acados_template/acados_ocp_solver_pyx.pyx \
	-I /root/robotic-mpc/code_gen/rated_code_ocp_aa12ba10 \

ocp_cython_o: ocp_cython_c
	$(CC) $(ACADOS_FLAGS) -c -O2 \
	-fPIC \
	-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION \
	-o acados_ocp_solver_pyx.o \
	-I $(INCLUDE_PATH)/blasfeo/include/ \
	-I $(INCLUDE_PATH)/hpipm/include/ \
	-I $(INCLUDE_PATH) \
	-I /root/miniforge3/lib/python3.12/site-packages/numpy/_core/include \
	-I /root/miniforge3/include/python3.12 \
	acados_ocp_solver_pyx.c \

ocp_cython: ocp_cython_o
	$(CC) $(ACADOS_FLAGS) -shared \
	-o acados_ocp_solver_pyx.so \
	-Wl,-rpath=$(LIB_PATH) \
	acados_ocp_solver_pyx.o \
	$(abspath .)/libacados_ocp_solver_six_dof_robot_aa12ba10.so \
	$(LDFLAGS) $(LDLIBS)




clean:
	$(RM) $(OBJ) $(EX_OBJ)
	$(RM) $(LIBACADOS_SOLVER) $(LIBACADOS_OCP_SOLVER)
	$(RM) $(EX_EXE)
clean_ocp_shared_lib:
	$(RM) $(LIBACADOS_OCP_SOLVER)
	$(RM) $(OCP_OBJ)

clean_ocp_cython:
	$(RM) libacados_ocp_solver_six_dof_robot_aa12ba10.so
	$(RM) acados_solver_six_dof_robot_aa12ba10.o
	$(RM) acados_ocp_solver_pyx.so
	$(RM) acados_ocp_solver_pyx.o


