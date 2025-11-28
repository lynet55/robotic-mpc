
from turtle import delay, position
import casadi as ca
import numpy as np
import math
import time
from trajectory_optimizer import MPC as model_predictive_control
from loader import UrdfLoader as urdf
from visualizer import MeshCatVisualizer as robot_visualizer
from prediction_model import SixDofRobot as six_dof_model
from surface import Surface

def run_sim(scene, model, mpc, Nsim, x0):
    """
    Run the closed-loop simulation.

    Parameters
    ----------
    scene : MeshCatVisualizer
        Visualizer instance used to render the robot.
    model : SixDofRobot
        Robot model providing dynamics and kinematics.
    """

    N_horizon = mpc.N_horizon
    Tf = mpc.Tf

    solver = mpc.solver
    integrator = mpc.setup_integrator(Tf/N_horizon)

    nx = solver.acados_ocp.dims.nx
    nu = solver.acados_ocp.dims.nu
    n_dof = model.n_dof

    simX = np.zeros((Nsim+1, nx))
    simU = np.zeros((Nsim, nu))
    pose_ee_euler = np.zeros((Nsim+1, 6)) 
    vee = np.zeros((Nsim+1, 6)) 

    simX[0,:] = x0
    pose_ee_euler[0] = np.array(model.forward_kinematics_euler(simX[0,:n_dof])).flatten()
    vee[0] = np.array(model.differential_kinematics(simX[0,:n_dof], simX[0,n_dof:])).flatten()

    t = np.zeros((Nsim)) 

    traj_points = [pose_ee_euler[0, :3].copy()]
    traj_array = np.vstack(traj_points)



    # Scene visualization
    
    scene.add_line(
        traj_array,
        path="lines/ee_trajectory",
        color=0xFF0000,
        line_width=2.0,
    )
    
    scene.add_triad(
        position=pose_ee_euler[0, :3],
        orientation_rpy=pose_ee_euler[0, 3:6],
        path="frames/end_effector_frame",
        scale=0.2,
        line_width=2.0
    )

    for i in range(Nsim):
        # solve ocp and get next control input
        simU[i,:] = solver.solve_for_x0(x0_bar = simX[i, :])

        t[i] = solver.get_stats('time_tot')

        # simulate system
        simX[i+1, :] = integrator.simulate(x=simX[i, :], u=simU[i,:])

        q1, q2, q3, q4, q5, q6 = simX[i+1,:n_dof]

        #Visual Update
        scene.set_joint_angles({
            "shoulder_pan_joint": q1,
            "shoulder_lift_joint": q2,
            "elbow_joint": q3,
            "wrist_1_joint": q4,
            "wrist_2_joint": q5,
            "wrist_3_joint": q6
        })

        pose_ee_euler[i+1] = np.array(model.forward_kinematics_euler(simX[i+1,:n_dof])).flatten()
        vee[i+1] = np.array(model.differential_kinematics(simX[i+1,:n_dof], simX[i+1,n_dof:])).flatten()

        traj_points.append(pose_ee_euler[i+1,:3].copy())
        traj_array = np.vstack(traj_points)

        scene.update_triad("frames/end_effector_frame", position=pose_ee_euler[i+1][:3], orientation_rpy=pose_ee_euler[i+1][3:6])
        scene.update_line(path="lines/ee_trajectory",points=traj_array,)

    
    print("\nSimulation complete!")

  

if __name__ == "__main__":

    Ts = 0.001      # 10 ms
    N  = 200
    T  = N*Ts  # orizzonte di 0.4 secondi
    Nsim = 4000

    surface_limits = ((-2, 2), (-2, 2))
    surface_origin = np.array([0.0, 0.0, 0.0])
    surface_orientation_rpy = np.array([0.0, 0.0, 0.0])

    qdot_0 = np.array([2,2,2,2,2,2], dtype=np.float64)
    q_0 = np.array([
    0.0,             # q1: base
    -np.pi/3,        # q2: spalla giù (braccio che si alza verso l'alto/z+)
    np.pi/3,         # q3: gomito in avanti
    -np.pi/2,        # q4: polso perpendicolare alla superficie
    -np.pi/2,        # q5: orienta il tool verso il basso
    0.0              # q6: rotazione attorno all'asse del tool
    ])
    Wcv = np.array([5,5,5,5,5,5], dtype=np.float64)

    robot_loader = urdf("ur5")
    scene = robot_visualizer(robot_loader)
    surface = Surface(
        position=surface_origin,
        orientation_rpy=surface_orientation_rpy,
        limits=surface_limits
    )
    robot = six_dof_model(
        urdf_loader=robot_loader,
        Wcv=Wcv
    )

    x0 = np.hstack((q_0, qdot_0))

    mpc = model_predictive_control(
        surface=surface,
        state=robot.state,
        initial_state=x0,
        control_input=robot.input,
        model=robot,
        forward_kinematics=robot.fk_casadi_rot, 
        differential_kinematics=robot.dk_casadi,
        N_horizon=N,
        Tf=T
    )
    
    scene.add_surface_from_casadi(
        surface.get_surface_function(),
        x_limits=surface.get_limits()[0],
        y_limits=surface.get_limits()[1],
        resolution=80,
        path="surfaces/quadratic_surface",
        color=0x3399FF,
        opacity=0.6,    
        origin=surface_origin,            
        orientation_rpy=surface_orientation_rpy,    
    )
    
    run_sim(
        scene,
        robot,
        mpc,
        Nsim,
        x0,
    )