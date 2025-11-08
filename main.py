from acados import Acados as mpc
import casadi as ca


def run_sim(controller, solver, surface, n_timesteps):


    for i in range(n_timesteps):

        solver.set(0, 'lbx', x_current)
        solver.set(0, 'ubx', x_current)
        
        solver.solve()
        u_opt = solver.get(0, "u")

  

if __name__ == "_main_":

    x = ca.SX.sym("x")
    y = ca.SX.sym("y")
    surface = 2*x**2 + 2*y**2 + 2*x*y + 2*x + 2*y + 2

    #States
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    z = ca.SX.sym('z')
    vx = ca.SX.sym('vx')
    vy = ca.SX.sym('vy')
    vz = ca.SX.sym('vz')
    state = ca.vertcat(x, y, z, vx, vy, vz)

    # Control inputs
    ax = ca.SX.sym('ax')
    ay = ca.SX.sym('ay')
    az = ca.SX.sym('az')
    control = ca.vertcat(ax, ay, az)

    controller = mpc(
        surface = surface,
        state = state,
        control_input = control,
        explicit_model= ur5.get_explicit_model(),
        implisit_model= ur5.get_implcit_model()
    )

    run_sim(controller, n_timesteps = 100)