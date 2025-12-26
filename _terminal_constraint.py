import numpy as np
import matplotlib.pyplot as plt
import yaml
from scipy.linalg import solve_discrete_are
from models.prediction_model import SixDofRobot
from Infrastructure.loader import UrdfLoader as urdf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "mpc.yaml"


def export_lqr_terminal_to_yaml(
    yaml_path: Path,
    P: np.ndarray,
    alpha: float,
    diag_tol: float = 1e-10
):


    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with yaml_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault("lqr_terminal", {})

    # check if P is (numerically) diagonal
    off_diag_norm = np.linalg.norm(P - np.diag(np.diag(P)))
    if off_diag_norm < diag_tol:
        cfg["lqr_terminal"]["P"] = {
            "type": "diagonal",
            "values": np.diag(P).tolist()
        }
    else:
        cfg["lqr_terminal"]["P"] = {
            "type": "full",
            "values": P.tolist()
        }

    cfg["lqr_terminal"]["alpha"] = float(alpha)

    with yaml_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print("[YAML] Saved compact LQR terminal set")


def discrete_lqr(A, B, Q, R):
    """Discrete-time LQR: returns K, P solving DARE."""
    P = solve_discrete_are(A, B, Q, R)
    K = -np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


def build_constraints(q_min, q_max, qdot_min, qdot_max, nx=12, nu=6):
    assert nx == 12 and nu == 6, "Expected dimensions: nx=12, nu=6"

    E = np.block([np.eye(nx // 2), np.zeros((nx // 2, nx // 2))])  # (6,12)

    Cx = np.vstack([E, -E])              # (12,12)
    dx = np.hstack([q_max, -q_min])      # (12,)

    Cu = np.vstack([np.eye(nu), -np.eye(nu)])  # (12,6)
    du = np.hstack([qdot_max, -qdot_min])      # (12,)

    return Cx, dx, Cu, du


def alpha_from_constraints_over_ellipsoide(P, C, d, safety=0.98):
    Pinv = np.linalg.inv(P)

    alpha_i = np.empty(len(d), dtype=float)
    for i in range(len(d)):
        ci = C[i, :].reshape(-1, 1)
        denom = float((ci.T @ Pinv @ ci).item())
        if denom <= 1e-15:
            alpha_i[i] = np.inf if d[i] >= 0 else 0.0
        else:
            alpha_i[i] = (d[i] ** 2) / denom

    idx_active = int(np.argmin(alpha_i))
    alpha_star = float(np.min(alpha_i))
    return safety * alpha_star, alpha_i, idx_active


def ellipsoid_metric_J(P, alpha):
    n = P.shape[0]
    detP = np.linalg.det(P)
    if detP <= 0 or alpha <= 0:
        return np.nan
    return (alpha ** (n / 2.0)) / np.sqrt(detP)


def terminal_ellipsoid_from_lqr(A_d, B_d, Q, rho, Cx, dx, Cu, du, safety=0.98):
    nu = B_d.shape[1]
    R = float(rho) * np.eye(nu)

    K, P = discrete_lqr(A_d, B_d, Q, R)

    C = np.vstack([Cx, Cu @ K])  # (24,12)
    d = np.hstack([dx, du])      # (24,)

    alpha, alpha_i, idx_active = alpha_from_constraints_over_ellipsoide(P, C, d, safety=safety)
    J = ellipsoid_metric_J(P, alpha)

    slack = alpha_i / max(alpha, 1e-18)

    return {
        "K": K, "P": P,
        "alpha": alpha, "alpha_i": alpha_i, "idx_active": idx_active,
        "J": J, "slack": slack,
        "C": C, "d": d,
        "P": P
    }


def plot_J_vs_rho(rho_grid, J_list, idx_active_list, nu, rho_star):
    idx_active_list = np.asarray(idx_active_list, dtype=int)
    is_state_limited = (idx_active_list < 2 * nu)

    J_state = np.where(is_state_limited, J_list, np.nan)
    J_input = np.where(~is_state_limited, J_list, np.nan)

    plt.figure()
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(rho_grid, J_state, 'o-', label="State-limited (q bounds)")
    plt.plot(rho_grid, J_input, 'o-', label="Input-limited (u=Kx bounds)")

    plt.axvline(rho_star, linestyle='--', linewidth=2, label=fr"Selected $\rho^\star={rho_star:.2e}$")
    sel_k = int(np.argmin(np.abs(rho_grid - rho_star)))
    plt.plot([rho_star], [J_list[sel_k]], marker='o', markersize=10, linestyle='None')

    plt.xlabel(r"$\rho$ in $R=\rho I$")
    plt.ylabel(r"$J(P,\alpha)=\alpha^{n/2}/\sqrt{\det(P)}$")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

def fmt_sci_latex(x, sig=3):
    """
    Return x formatted as LaTeX scientific notation: a \\times 10^{b}.
    """
    if x == 0 or not np.isfinite(x):
        return rf"{x:.{sig}g}"
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / (10 ** exp)
    return rf"{mant:.{sig}f}\times 10^{{{exp:d}}}"


def plot_state_slack_comparison(slack_base, slack_tuned, J_base, J_tuned, nu):
    """
    Paper-style dumbbell plot for state constraint slacks before/after Q tuning.
    Color scheme aligned with R-tuning plot (single blue tone).
    """

    s_base = np.asarray(slack_base[:2*nu], dtype=float)
    s_tuned = np.asarray(slack_tuned[:2*nu], dtype=float)

    labels = []
    for j in range(nu):
        labels += [rf"$q_{{{j+1},\min}}$", rf"$q_{{{j+1},\max}}$"]
    x = np.arange(2*nu)

    Jb = fmt_sci_latex(J_base, sig=3)
    Jt = fmt_sci_latex(J_tuned, sig=3)

    blue = "tab:blue"

    plt.figure(figsize=(10.5, 4.2))

    # Dumbbell segments (light blue)
    for i in range(2*nu):
        plt.plot(
            [x[i], x[i]],
            [s_base[i], s_tuned[i]],
            color=blue,
            alpha=0.25,
            linewidth=1.2,
            zorder=1
        )

    # Baseline Q: open blue circles
    plt.scatter(
        x, s_base,
        s=75,
        facecolors='none',
        edgecolors=blue,
        linewidths=1.4,
        marker='o',
        label=rf"Baseline $Q$  ($\mathbf{{J}}={Jb}$)",
        zorder=3
    )

    # Tuned Q: filled blue circles
    plt.scatter(
        x, s_tuned,
        s=75,
        color=blue,
        marker='o',
        label=rf"Tuned $Q$  ($\mathbf{{J}}={Jt}$)",
        zorder=4
    )

    # Active constraint (minimum tuned slack)
    i_active = int(np.argmin(s_tuned))
    plt.scatter(
        [x[i_active]],
        [s_tuned[i_active]],
        s=75,
        facecolors='none',
        edgecolors='k',
        linewidths=1.6,
        zorder=6
    )

    plt.annotate(
        rf"active constraint: {labels[i_active]}",
        xy=(x[i_active], s_tuned[i_active]),
        xytext=(x[i_active] + 0.5, s_tuned[i_active] + 0.18),
        arrowprops=dict(arrowstyle="->", lw=1.0),
        fontsize=12
    )

    # Reference line
    plt.axhline(1.0, linestyle='--', linewidth=1.4, color='k')

    plt.xticks(x, labels, fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylim(0.95, max(2.5, 1.05*np.max([s_base.max(), s_tuned.max()])))

    plt.ylabel(r"Slack ratio $s_i=\alpha_i/\alpha$", fontsize=13)
    plt.xlabel("Joint position constraints", fontsize=13)
    '''
    plt.title(
        "Effect of $Q$ tuning on joint position constraint utilization",
        fontsize=14
    )'''

    plt.grid(True, axis='y', linestyle=':', linewidth=0.8, alpha=0.8)
    plt.legend(fontsize=11, frameon=False)
    plt.tight_layout()





def tune_Q_diagonal_for_state_slacks(
    A_d, B_d,
    Q_diag_init,
    rho_star,
    Cx, dx, Cu, du,
    nu,
    safety=0.98,
    n_iter=12,
    beta=0.35,
    w_min=1e-3,
    w_max=1e6
):
    Q_diag = np.array(Q_diag_init, dtype=float).copy()

    hist = []
    for it in range(n_iter):
        Q = np.diag(Q_diag)
        res = terminal_ellipsoid_from_lqr(A_d, B_d, Q, rho_star, Cx, dx, Cu, du, safety=safety)

        slack = res["slack"]
        slack_state = slack[:2*nu]

        s_plus = slack_state[:nu]
        s_minus = slack_state[nu:2*nu]
        s_joint = np.maximum(s_plus, s_minus)

        wq = Q_diag[:nu]
        wq_new = wq * np.power(np.maximum(s_joint, 1.0), -beta)
        wq_new = np.clip(wq_new, w_min, w_max)

        # keep geometric mean (avoid global drift)
        g_old = np.exp(np.mean(np.log(np.maximum(wq, 1e-18))))
        g_new = np.exp(np.mean(np.log(np.maximum(wq_new, 1e-18))))
        if g_new > 0:
            wq_new *= (g_old / g_new)

        Q_diag[:nu] = wq_new

        hist.append({
            "iter": it,
            "Q_diag": Q_diag.copy(),
            "alpha": res["alpha"],
            "J": res["J"],
            "idx_active": res["idx_active"],
            "slack": slack.copy(),
            "slack_state": slack_state.copy(),
            "s_joint": s_joint.copy()
        })

    return Q_diag, hist


def main():
    robot_loader = urdf('ur5')
    prediction_model = SixDofRobot(
        urdf_loader=robot_loader,
        Ts=0.0005,
        Wcv=np.array([228.9, 262.09, 517.3, 747.44, 429.9, 1547.76], dtype=float)
    )

    A_d = prediction_model.Ad
    B_d = prediction_model.Bd
    nu = int(prediction_model.n_dof)
    nx = 2 * nu

    q_min = np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi], dtype=float)
    q_max = np.array([+2*np.pi, +2*np.pi, +np.pi, +2*np.pi, +2*np.pi, +2*np.pi], dtype=float)
    qdot_min = -np.pi * np.ones(nu, dtype=float)
    qdot_max =  np.pi * np.ones(nu, dtype=float)

    Cx, dx, Cu, du = build_constraints(q_min, q_max, qdot_min, qdot_max, nx=nx, nu=nu)

    # Baseline Q
    wq = 10.0
    wqd = 1.0
    Wq = np.full(nu, wq, dtype=float)
    Wq[2] *= 10.0
    Q_diag0 = np.hstack([Wq, np.full(nu, wqd, dtype=float)])
    Q0 = np.diag(Q_diag0)

    # ==========================
    # (1) R tuning via rho grid
    # ==========================
    rho_grid = np.logspace(0, 4, 60)
    J_list = np.zeros_like(rho_grid, dtype=float)
    idx_active_list = np.zeros_like(rho_grid, dtype=int)

    for k, rho in enumerate(rho_grid):
        res = terminal_ellipsoid_from_lqr(A_d, B_d, Q0, rho, Cx, dx, Cu, du, safety=0.98)
        J_list[k] = res["J"]
        idx_active_list[k] = res["idx_active"]

    valid = np.isfinite(J_list)
    if not np.any(valid):
        raise RuntimeError("All J are NaN/inf; check P positivity / constraints / DARE feasibility.")

    rho_star = float(rho_grid[valid][np.argmax(J_list[valid])])
    print(f"[R-TUNING] selected rho* (max J) = {rho_star:.6e}")
    plot_J_vs_rho(rho_grid, J_list, idx_active_list, nu, rho_star)

    # ==========================
    # (2) Q tuning
    # ==========================
    res_base = terminal_ellipsoid_from_lqr(A_d, B_d, Q0, rho_star, Cx, dx, Cu, du, safety=0.98)
    print(f"[BASE] alpha={res_base['alpha']:.3e}, J={res_base['J']:.3e}, idx_active={res_base['idx_active']}")

    Q_diag_tuned, hist = tune_Q_diagonal_for_state_slacks(
        A_d, B_d,
        Q_diag_init=Q_diag0,
        rho_star=rho_star,
        Cx=Cx, dx=dx, Cu=Cu, du=du,
        nu=nu,
        safety=0.98,
        n_iter=8,
        beta=0.35
    )

    Q_tuned = np.diag(Q_diag_tuned)
    res_tuned = terminal_ellipsoid_from_lqr(A_d, B_d, Q_tuned, rho_star, Cx, dx, Cu, du, safety=0.98)
    print(f"[TUNED] alpha={res_tuned['alpha']:.3e}, J={res_tuned['J']:.3e}, idx_active={res_tuned['idx_active']}")

    export_lqr_terminal_to_yaml(
    yaml_path=CONFIG_PATH,
    P=res_tuned["P"],
    alpha=res_tuned["alpha"]
    )



    # Plot A: slack comparison with J in legend
    plot_state_slack_comparison(
        slack_base=res_base["slack"],
        slack_tuned=res_tuned["slack"],
        J_base=res_base["J"],
        J_tuned=res_tuned["J"],
        nu=nu
    )

    plt.show()


if __name__ == "__main__":
    main()
