import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are
from prediction_model import SixDofRobot
from loader import UrdfLoader as urdf

def discrete_lqr(A, B, Q, R):
    """
    Discrete-time LQR.
    Returns K, P by solving DARE.
    """
    P = solve_discrete_are(A, B, Q, R)
    K = -np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P

def build_constraints(q_min, q_max, qdot_min, qdot_max, nx=12, nu=6):
    """
    Build (Cx, dx) for joint position constraints on q = E x,
    and (Cu, du) for input bounds u.
    """
    assert nx == 12 and nu == 6, "Expected dimensions: nx = 12, nu = 6"

    if (q_min.shape != (nx//2,) or q_max.shape != (nx//2,) or
        qdot_min.shape != (nu,) or qdot_max.shape != (nu,)):
        raise ValueError(
            f"""Expected shapes:
            - q_min, q_max: {(nx//2,)}
            - qdot_min, qdot_max: {(nu,)}
            """
        )

    # E extracts q from x = [q; qdot]
    E = np.block([np.eye(nx//2), np.zeros((nx//2, nx//2))])  

    Cx = np.vstack([E, -E])          # (2nx, nx)                       
    dx = np.hstack([q_max, -q_min])  # (nx, nx)            

    Cu = np.vstack([np.eye(6), -np.eye(6)])       
    du = np.hstack([qdot_max, -qdot_min])         

    return Cx, dx, Cu, du

def alpha_from_ellipsoid_constraints(P, C, d, safety=0.98):
    """
    Compute alpha such that {x: x^T P x <= alpha} subset {x: Cx <= d}.
    For each constraint i: alpha <= d_i^2 / (c_i^T P^{-1} c_i).
    safety < 1 shrinks alpha slightly for numerical robustness.
    """
    Pinv = np.linalg.inv(P)

    alpha_i = np.empty(len(d), dtype=float)
    for i in range(len(d)):
        ci = C[i, :].reshape(-1, 1)  # column
        denom = (ci.T @ Pinv @ ci).item()
        if denom <= 1e-15:
            alpha_i[i] = np.inf if d[i] >= 0 else 0.0
        else:
            alpha_i[i] = (d[i] ** 2) / denom

    alpha_star = float(np.min(alpha_i))
    idx_active = int(np.argmin(alpha_i))
    return safety * alpha_star, alpha_i, idx_active

def plot_alpha_limits(alpha_i, idx_active, labels=None, title="Constraint-induced alpha limits"):
    plt.figure()
    plt.plot(alpha_i, marker='o', linestyle='None')
    plt.plot(idx_active, alpha_i[idx_active], marker='o', markersize=10)
    plt.yscale("log")
    plt.xlabel("Constraint index i")
    plt.ylabel(r"$\alpha_i = d_i^2 / (c_i^\top P^{-1} c_i)$")
    plt.title(title)
    if labels is not None:
        # show a few labels if provided
        pass
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

def plot_2d_slice(P, alpha, q_min, q_max, dims=(0, 1), title="2D slice (others = 0)"):
    """
    Illustrative 2D slice of ellipsoid x^T P x <= alpha for x[dims] varying, others=0.
    This is NOT the exact projection; it's a slice. Useful to visually sanity-check.
    """
    # index of coordinates to consider 
    i, j = dims 
    # Extract 2x2 block for the slice
    P2 = P[np.ix_([i, j], [i, j])]

    # Parametric ellipse boundary: z^T P2 z = alpha
    # Use eigen-decomposition of P2: P2 = V diag(lam) V^T
    lam, V = np.linalg.eigh(P2)
    # z = V diag(1/sqrt(lam)) * sqrt(alpha) * [cos t; sin t]
    t = np.linspace(0, 2*np.pi, 400)
    circ = np.vstack([np.cos(t), np.sin(t)])
    scale = np.diag(np.sqrt(alpha) / np.sqrt(lam))
    ell = (V @ scale @ circ).T  # (N,2)

    plt.figure()
    plt.plot(ell[:,0], ell[:,1])
    if i < 6 and j < 6:
        plt.axvline(q_min[i], linestyle="--")
        plt.axvline(q_max[i], linestyle="--")
        plt.axhline(q_min[j], linestyle="--")
        plt.axhline(q_max[j], linestyle="--")
    plt.xlabel(fr"$x_{i}$")
    plt.ylabel(fr"$x_{j}$")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)

def main():
    
    # Definition of LTI discrete matrices 
    robot_loader = urdf('ur5')
    prediction_model = SixDofRobot(
            urdf_loader=robot_loader,
            Ts=0.0005,
            Wcv=np.array([228.9,262.09,517.3,747.44,429.9,1547.76])
        )
    
    A_d = prediction_model.Ad
    B_d = prediction_model.Bd

    nx = 2*prediction_model.n_dof
    nu = prediction_model.n_dof

    # LQR weight matrices
    wq = 1.0
    wqd = 0.1
    Q = np.diag([wq]*nu + [wqd]*nu)
    R = 0.1*np.eye(nu)


    q_min = np.array([-2*np.pi,-2*np.pi,-np.pi,-2*np.pi,-2*np.pi,-2*np.pi], dtype=np.float64)
    q_max = np.array([-2*np.pi,-2*np.pi,-np.pi,-2*np.pi,-2*np.pi,-2*np.pi], dtype=np.float64)
    qdot_min = -np.pi * np.ones(nu)
    qdot_max =  np.pi * np.ones(nu)

    # LQR matrices from ricatti equations
    K, P = discrete_lqr(A_d, B_d, Q, R)
    print(f"P dimensions: {np.shape(P)}")


    # Construction of linear inequalities
    Cx, dx, Cu, du = build_constraints(q_min, q_max, qdot_min, qdot_max, nx=nx, nu=nu)

    # Linear state inequalities over LQR auxiloary control law
    C = np.vstack([Cx, Cu @ K])
    d = np.hstack([dx, du])

    # Derivation of alpha
    alpha, alpha_i, idx_active = alpha_from_ellipsoid_constraints(P, C, d, safety=0.98)

    print(f"alpha* (with safety) = {alpha:.6e}")
    print(f"active constraint index = {idx_active}")
    print(f"raw alpha_active = {alpha_i[idx_active]:.6e}")
    print(f"LQR gain K: {K}")

    # -----------------------------
    # 7) Plot diagnostici
    # -----------------------------
    plot_alpha_limits(alpha_i, idx_active, title="Alpha limits from all constraints (log scale)")

    # Slice 2D sulle prime due posizioni (q1,q2): indices 0,1
    plot_2d_slice(P, alpha, q_min=q_min,
                  q_max=q_max,
                  dims=(2, 10),
                  title="Ellipsoid slice on (q1,q2), others=0, with joint position bounds")

    plt.show()

if __name__ == "__main__":
    main()
