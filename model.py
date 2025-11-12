import numpy as np

class SixDofRobot:
    def __init__(self, initial_state, urdf_loader, integration_method="RK2"):
        self._model = urdf_loader.model
        self._data = urdf_loader.data
        self._z0 = initial_state
        self._integration_method_name = integration_method
        
        if integration_method == "RK2":
            self._integration_method = self._RK2
        elif integration_method == "RK4":
            self._integration_method = self._RK4
        elif integration_method == "Euler":
            self._integration_method = self._forward_euler_integration
        else:
            raise ValueError(f"Unknown integration method: {integration_method}")
    
    def update(self, state, dt):
        return self._integration_method(state, dt)
    
    def _forward_euler_integration(self, state, dt):
        state_dot = self._compute_dynamics(state)
        new_state = state + dt * state_dot
        return new_state
    
    def _RK2(self, state, dt):
        k1 = self._compute_dynamics(state)
        state_mid = state + 0.5 * dt * k1
        k2 = self._compute_dynamics(state_mid)
        new_state = state + dt * k2
        return new_state
    
    def _RK4(self, state, dt):
        k1 = self._compute_dynamics(state)
        k2 = self._compute_dynamics(state + 0.5 * dt * k1)
        k3 = self._compute_dynamics(state + 0.5 * dt * k2)
        k4 = self._compute_dynamics(state + dt * k3)
        new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return new_state
    
    def _compute_dynamics(self, state):
        n_dof = len(state) // 2
        q = state[:n_dof]
        q_dot = state[n_dof:]
        q_ddot = self._compute_forward_dynamics(q, q_dot)
        state_dot = np.concatenate([q_dot, q_ddot])
        return state_dot
    
    def _compute_forward_dynamics(self, q, q_dot):
        return np.zeros_like(q)