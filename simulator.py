from loader import UrdfLoader as urdf
from visualizer import MeshCatVisualizer as robot_visualizer
from prediction_model import SixDofRobot as prediction_robot_6dof
from simulation_model import Robot as simulation_robot_6dof
from trajectory_optimizer import MPC as model_predictive_control
from surface import Surface
import numpy as np

class Simulator:
    def __init__(self, dt, simulation_time, prediction_horizon, surface_limits, surface_origin, surface_orientation_rpy, qdot_0, q_0, wcv, scene=True):
        self.dt = dt
        self.Nsim = int(simulation_time/dt)  # Total number of simulation steps
        self.prediction_horizon = prediction_horizon  # Number of steps in MPC horizon
        self.surface_limits = surface_limits
        self.surface_origin = surface_origin
        self.surface_orientation_rpy = surface_orientation_rpy
        
        self.qdot_0 = qdot_0
        self.q_0 = q_0
        self.wcv = wcv

        self.robot_loader = urdf('ur5')

        if scene:
            self.scene = robot_visualizer(self.robot_loader)
        else:
            self.scene = None

        self.surface = Surface(
            position=surface_origin,
            orientation_rpy=surface_orientation_rpy,
            limits=surface_limits
        )
        self.simulation_model = simulation_robot_6dof(
            urdf_loader=self.robot_loader,
            z0=np.hstack((self.q_0, self.qdot_0)),
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
            state=self.prediction_model.state,
            initial_state=np.hstack((self.q_0, self.qdot_0)),
            control_input=self.prediction_model.input,
            model=self.prediction_model,
            forward_kinematics=self.prediction_model.fk_casadi_rot, 
            differential_kinematics=self.prediction_model.dk_casadi,
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
            origin=surface_origin,            
            orientation_rpy=surface_orientation_rpy,    
        )

        self.scene.add_triad(
            position=self.simulation_model.ee_position(0),
            orientation_rpy=self.simulation_model.ee_orientation(0),
            path="frames/end_effector_frame",
            scale=0.2,
            line_width=2.0
        )

    def run(self):

        for t in range(self.Nsim):
            #MPC input and refferences
            current_state = self.simulation_model.state(t)        
            self.mpc.solver.set(0, 'lbx', current_state)
            self.mpc.solver.set(0, 'ubx', current_state)
            status = self.mpc.solver.solve()
            u = self.mpc.solver.get(0, "u")

            #Integration Step
            self.simulation_model.update(self.simulation_model.state(t), u, t)
            q1, q2, q3, q4, q5, q6 = self.simulation_model.joint_angles(t)

            #Visual Update
            if self.scene is not None:
                self.scene.set_joint_angles({
                "shoulder_pan_joint": q1,
                "shoulder_lift_joint": q2,
                "elbow_joint": q3,
                "wrist_1_joint": q4,
                "wrist_2_joint": q5,
                "wrist_3_joint": q6
                })

                self.scene.update_triad("frames/end_effector_frame", position=self.simulation_model.ee_position(t+1), orientation_rpy=self.simulation_model.ee_orientation(t+1))
    
if __name__ == "__main__":
    sim0 = Simulator(
        dt=0.001,
        prediction_horizon=200,
        simulation_time=4000,
        surface_limits=((-2, 2), (-2, 2)),
        surface_origin=np.array([0.0, 0.0, 0.0]),
        surface_orientation_rpy=np.array([0.0, 0.0, 0.0]),
        qdot_0=np.array([2,2,2,2,2,2]),
        q_0=np.array([0.0, -np.pi/3, np.pi/3, -np.pi/2, -np.pi/2, 0.0]),
        wcv=np.array([5,5,5,5,5,5]),
        scene=True
    )
    sim0.run()
    # sim0_solver = sim0.opc_solver.stats
    # sim0_simulation_results = sim0.simulation_model.stats
    # print(sim0.benchmark)