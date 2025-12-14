import yaml
import numpy as np
import matplotlib.pyplot as plt
from model import Robot
from loader import UrdfLoader

# PARAMETERS 
T  = 4.0
Tu = 4.0
u_step = 0.1

q_0 = np.zeros(6, dtype=np.float64)
qdot_0 = np.zeros(6, dtype=np.float64)
z_0 = np.hstack((q_0, qdot_0))

w_cv = np.array([5, 10, 20, 30, 40, 50], dtype=np.float64)

def analyze_global_error():
    urdf = UrdfLoader('ur5_robot')

    N_list = np.logspace(1, 5, num=9, dtype=int)  
    Ts_list = []

    M_euler, err_euler  = np.zeros(len(N_list), dtype=float), np.zeros(len(N_list), dtype=float)
    M_rk2,   err_rk2    = np.zeros(len(N_list), dtype=float), np.zeros(len(N_list), dtype=float)
    M_rk4,   err_rk4    = np.zeros(len(N_list), dtype=float), np.zeros(len(N_list), dtype=float)

    robot = Robot(1, T, Tu, z_0, u_step, urdf)

    for i, N in enumerate(N_list):
        Ts = float(T / N)
        Ts_list.append(Ts)

        print(f"\n--- N = {N}, Ts = {Ts:.3e} ---")

        robot.Ts = Ts

        # Lagrange Formula
        z_exact, _ = robot.lagrange_formula(w_cv)
        zT_exact = z_exact[:, -1]

        # Euler
        z_euler, _ = robot.euler_simulation(w_cv)
        zT_euler = z_euler[:, -1]
        err = np.linalg.norm(zT_euler - zT_exact)
        M_euler[i] = 1 * N    # s = 1
        err_euler[i] = err

        # RK2
        z_rk2, _ = robot.rk2_mp_simulation(w_cv)
        zT_rk2 = z_rk2[:, -1]
        err = np.linalg.norm(zT_rk2 - zT_exact)
        M_rk2[i] = 2 * N     # s = 2
        err_rk2[i] = err

        # RK4
        z_rk4, _ = robot.rk4_simulation(w_cv)
        zT_rk4 = z_rk4[:, -1]
        err = np.linalg.norm(zT_rk4 - zT_exact)
        M_rk4[i] = 4 * N      # s = 4
        err_rk4[i] = err

    fig, ax = plt.subplots()

    ax.loglog(M_euler, err_euler, 'o-', label='Explicit Euler')
    ax.loglog(M_rk2,   err_rk2,   's-', label='RK2 midpoint')
    ax.loglog(M_rk4,   err_rk4,   'd-', label='RK4')

    ax.set_xlabel('function evaluations')
    ax.set_ylabel(f'E(T),  T={T}')
    ax.grid(True, which='both', linestyle=':')
    ax.legend()
    ax.set_title('Global error vs function evaluations')

    def fe_to_Ts(M):
        return T / M

    def Ts_to_fe(Ts):
        return T / Ts

    secax = ax.secondary_xaxis('top', functions=(fe_to_Ts, Ts_to_fe))
    secax.set_xlabel(r'sampling time $T_s$ [s]')

    secax.set_xticks(Ts_list)
    secax.set_xticklabels([f"{Ts:.1e}" for Ts in Ts_list])

    plt.show()



if __name__ == "__main__":
    analyze_global_error()

