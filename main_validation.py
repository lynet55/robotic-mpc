import yaml
import numpy as np
from model import Robot
from plot_visualization import Visualization
from loader import UrdfLoader

'''
with open('NOC-project/params.yaml', 'r') as file:
    params = yaml.safe_load(file)

Ts = params["Ts"]
...
Tu = params = params["Tu"]

'''

# PARAMETERS
Ts = 0.01
Tu = 4
Tsim = 4
Nsim = int(Tsim/Ts)
u_step = 1

q_0 = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
qdot_0 = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)

z_0 = np.hstack((q_0, qdot_0))
w_cv = np.array([228.9,262.09,517.3,747.44,429.9,1547.46], dtype=np.float64)


def run_sim():
    urdf = UrdfLoader('ur5_robot')
    robot = Robot(Ts,Tsim,Tu,z_0,u_step,urdf)

    z_euler, u = robot.euler_simulation(w_cv)
    z_rk2, _ = robot.rk2_mp_simulation(w_cv) 
    z_rk4, _ = robot.rk4_simulation(w_cv)
    z_exact, _ = robot.lagrange_formula(w_cv)

    y_euler = robot.compute_output(z_euler)
    y_rk2 = robot.compute_output(z_rk2)
    y_rk4 = robot.compute_output(z_rk4)
    y_exact = robot.compute_output(z_exact)

    t = np.linspace(0, T, N)
  
    # EULER INTEGRATION
    q_euler = [y for y in z_euler[:6]]
    labels_q_euler = [f" q{i+1} using euler integration" for i in range(6)]
    dq_euler = [y for y in z_euler[6:]]
    labels_dq_euler = [f" dq{i+1} using euler integration" for i in range(6)]

    
    # RK2 (middle-point) INTEGRATION
    q_rk2 = [y for y in z_rk2[:6]]
    labels_q_rk2 = [f" q{i+1} using rk2 integration" for i in range(6)]
    dq_rk2 = [y for y in z_rk2[6:]]
    labels_dq_rk2 = [f" dq{i+1} using rk2 integration" for i in range(6)]

    # RK4 INTEGRATION
    q_rk4 = [y for y in z_rk4[:6]]
    labels_q_rk4 = [f" q{i+1} using rk4 integration" for i in range(6)]
    dq_rk4 = [y for y in z_rk4[6:]]
    labels_dq_rk4 = [f" dq{i+1} using rk4 integration" for i in range(6)]
    
    # ANALITICAL RESPONCE 
    q_exact = [y for y in z_exact[:6]]
    labels_q_exact = [f" q{i+1} using lagrange formula" for i in range(6)]
    dq_exact = [y for y in z_exact[6:]]
    labels_dq_exact = [f" dq{i+1} using lagrange formula" for i in range(6)]
    
    # OUTPUT VIA z_euler
    p_euler = [pi for pi in y_euler[:3]]
    labels_p_euler = ["px_euler", "py_euler", "pz_euler"]
    ve_euler = [ve_i for ve_i in y_euler[7:10]]
    labels_ve_euler = ["vx_euler", "vy_euler", "vz_euler"]
    
    # OUTPUT VIA z_rk2
    p_rk2 = [pi for pi in y_rk2[:3]]
    labels_p_rk2 = ["px_rk2", "py_rk2", "pz_rk2"]
    ve_rk2 = [ve_i for ve_i in y_rk2[7:10]]
    labels_ve_rk2 = ["vx_rk2", "vy_rk2", "vz_rk2"]

    # OUTPUT VIA z_rk4
    p_rk4 = [pi for pi in y_rk4[:3]]
    labels_p_rk4 = ["px_rk4", "py_rk4", "pz_rk4"]
    ve_rk4 = [ve_i for ve_i in y_rk4[7:10]]
    labels_ve_rk4 = ["vx_rk4", "vy_rk4", "vz_rk4"]

    # OUTPUT VIA z_exact
    p_exact = [pi for pi in y_exact[:3]]
    labels_p_exact = ["px_exact", "py_exact", "pz_exact"]
    ve_exact = [ve_i for ve_i in y_exact[7:10]]
    labels_ve_exact = ["vx_exact", "vy_exact", "vz_exact"]
    


    '''
    joint_p1 = [q_euler[0], q_rk2[0], q_rk4[0], q_exact[0]]
    labels_jp1 = ["Joint 1 position with euler", "Joint 1 position with lagrange formula",
                    "Joint 1 position with rk2", "Joint 1 position with rk4"]
    '''
    vis = Visualization() 


    
    vis.plot_multiple_signals(t, dq_euler, labels_dq_euler,
                            title="Joint 1 position step response",
                            xlabel="time [s]",
                            ylabel="rotation [rad]")
    '''
    vis.plot_multiple_signals(t, joint_speeds, labels_js,
                            title="Joint velocities step response",
                            xlabel="time [s]",
                            ylabel="velocity [rad/s]")

    vis.plot_multiple_signals(t, p, labels_p,
                            title="End effector position step response",
                            xlabel="time [s]",
                            ylabel="position [m]")

    vis.plot_multiple_signals(t, p, labels_p,
                            title="End effector position step response",
                            xlabel="time [s]",
                            ylabel="position [m]")                      
    
    vis.plot_multiple_signals(t, joint_p1, labels_jp1,
                            title="End effector position step response",
                            xlabel="time [s]",
                            ylabel="position [m]")   
    
    vis.plot_3d_trajectory(p_euler[0], p_euler[1], p_euler[2])

    vis.animate_3d_trajectory(p_euler[0], p_euler[1], p_euler[2])
    '''
if __name__ == "__main__":
    run_sim()