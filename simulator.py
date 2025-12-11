from loader import UrdfLoader as urdf
from visualizer import MeshCatVisualizer as robot_visualizer
from prediction_model import SixDofRobot as prediction_robot_6dof
from simulation_model import Robot as simulation_robot_6dof
from trajectory_optimizer import MPC as model_predictive_control
from surface import Surface
from plotter import Plotter
import numpy as np
import time

class Simulator:
    def __init__(self, dt, simulation_time, prediction_horizon, surface_limits, surface_origin, surface_orientation_rpy, qdot_0, q_0, wcv, scene=True):
        self.dt = dt
        self.simulation_time = simulation_time
        self.Nsim = int(simulation_time/dt)  # Total number of simulation steps

        self.mpc_time = np.zeros(self.Nsim)
        self.integration_time = np.zeros(self.Nsim)
        
        self.prediction_horizon = prediction_horizon  # Number of steps in MPC horizon
        self.surface_limits = surface_limits
        self.surface_origin = surface_origin
        self.surface_orientation_rpy = surface_orientation_rpy
        
        self.qdot_0 = qdot_0
        self.q_0 = q_0
        self.wcv = wcv
        self.initial_state = np.hstack((self.q_0, self.qdot_0))

        self.robot_loader = urdf('ur5')
        self.scene = robot_visualizer(self.robot_loader)

        self.surface = Surface(
            position=surface_origin,
            orientation_rpy=surface_orientation_rpy,
            limits=surface_limits
        )
        self.simulation_model = simulation_robot_6dof(
            urdf_loader=self.robot_loader,
            z0=self.initial_state,
            u0=self.qdot_0,
            dt=self.dt,
            Nsim=self.Nsim,
            wcv=self.wcv,
            integration_method="RK4"
        )
        self.prediction_model = prediction_robot_6dof(
            urdf_loader=self.robot_loader,
            Wcv=self.wcv
        )

        self.mpc = model_predictive_control(
            surface=self.surface,
            initial_state=self.initial_state,
            model=self.prediction_model,
            N_horizon=self.prediction_horizon,
            Tf=self.dt*self.prediction_horizon
        )

        self.scene.add_surface_from_casadi(
            self.surface.get_surface_function(),
            x_limits=self.surface.get_limits()[0],
            y_limits=self.surface.get_limits()[1],
            resolution=80,
            path="surfaces/quadratic_surface",
            color=0x3399FF,
            opacity=0.6,    
            origin=self.surface_origin,            
            orientation_rpy=self.surface_orientation_rpy,    
        )

        ee_position_0 = self.simulation_model.ee_position(0)
        self.scene.add_triad(
            position=ee_position_0,
            orientation_rpy=self.simulation_model.ee_orientation(0),
            path="frames/end_effector_frame",
            scale=0.2,
            line_width=2.0
        )

        self.traj_points = [ee_position_0.copy()]
        self.traj_array = np.vstack(self.traj_points)

        self.scene.add_line(
            self.traj_array,
            path="lines/ee_trajectory",
            color=0xFF0000,
            line_width=3.0,
        )

    def run(self):

        for i in range(self.Nsim):
            start_time = time.time()
            #MPC input and refferences
            current_state = self.simulation_model.state(i)        

            mpc_start_time = time.time()
            self.mpc.solver.set(0, 'lbx', current_state)
            self.mpc.solver.set(0, 'ubx', current_state)
            self.mpc.solver.solve()
            u = self.mpc.solver.get(0, "u")
            mpc_time = time.time() - mpc_start_time
            self.mpc_time[i] = mpc_time

            #Integration Step
            integration_start_time = time.time()
            self.simulation_model.update(current_state, u, i)
            integration_time = time.time() - integration_start_time
            print(f"Integration time: {integration_time} s")
            self.integration_time[i] = integration_time

            #Visual Update
            if i % 10 == 0:
                q1, q2, q3, q4, q5, q6 = self.simulation_model.joint_angles(i+1)
                ee_pos = self.simulation_model.ee_position(i+1)
                
                self.scene.set_joint_angles({
                    "shoulder_pan_joint": q1,
                    "shoulder_lift_joint": q2,
                    "elbow_joint": q3,
                    "wrist_1_joint": q4,
                    "wrist_2_joint": q5,
                    "wrist_3_joint": q6
                })

                self.traj_points.append(ee_pos.copy())
                self.traj_array = np.vstack(self.traj_points)

                self.scene.update_triad("frames/end_effector_frame", position=ee_pos, orientation_rpy=self.simulation_model.ee_orientation(i+1))
                self.scene.update_line(path="lines/ee_trajectory", points=self.traj_array)
                print(f"Running time: {self.dt*i} s of {self.simulation_time} s")
            
            #realtime assurance
            time_spent = time.time() - start_time
            if time_spent < self.dt: time.sleep(self.dt - time_spent)

    def get_results(self):
        data = {
            'time': np.arange(0, self.simulation_model.z.shape[1]),
            'q': self.simulation_model.z[:6,:],
            'qdot': self.simulation_model.z[6:,:],
            'u': self.simulation_model.u,
            'ee_pose': self.simulation_model._ee_pose_log
        }
        return data

    
if __name__ == "__main__":
    sim0 = Simulator(
        dt=0.001,
        prediction_horizon=200,
        simulation_time=2,
        surface_limits=((-2, 2), (-2, 2)),
        surface_origin=np.array([0.0, 0.0, 0.0]),
        surface_orientation_rpy=np.array([0.0, 0.0, 0.0]),
        qdot_0=np.array([2,2,2,2,2,2]),
        q_0=np.array([np.pi/3, -np.pi/3, np.pi/4, -np.pi/2, -np.pi/2, 0.0]),
        wcv=np.array([5,10,15,20,25,35]),
        scene=True
    )
    sim0.run()
    results = sim0.get_results()
    plotter = Plotter()
    fig_joints = plotter.joint_angles(results['q'], dt=sim0.dt, title="Joint Angles")
    fig_timing = plotter.generic_plot(
        sim0.mpc_time,
        sim0.integration_time,
        dt=sim0.dt,
        xlabel="Time [s]",
        ylabel="Computation Time [s]",
        title="Timing Performance",
        labels=["MPC", "Integration"],
    )
    plotter.show(fig_joints)
    plotter.show(fig_timing)