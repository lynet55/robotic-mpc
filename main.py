import yaml

with open('NOC-project/params.yaml', 'r') as file:
    params = yaml.safe_load(file)

DH_params = params["UR5"]
sim = params = params["sim"]



# LOAD ROBOTICS MODEL
# LOAD REFFERECNE TRAJECTORY



def run_sim():

    # state storage

    #     // Cloosed loop simulation
    #     for n in size(trajectory):
    #         // STEP FORWARD TRAJECTORY
    #         // CONTROLS: ACADOS MPC SOLVER
    #         // MODEL: PLANT Robotics model
    #         // INTEGRATOR: Homemade RK4 probly
    #         // Set feedback


if __name__ == "__main__":
    run_sim()