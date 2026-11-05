from ...pid_control import PIDController
import numpy as np

class CascadedPIDControl:
    def __init__(self, control_gains: dict, dt:float) -> None:
        """Inits the PID controller object with control gains

        Args:
            control_gains (dict): contains all the control gains, packaged in a dict
            dt (float): sample time
        """ 
        self.roll_kp = control_gains['roll']['kp']
        self.pitch_kp = control_gains['pitch']['kp']
        self.yaw_kp = control_gains['yaw']['kp']

        self.p_kp = control_gains['p']['kp']
        self.p_ki = control_gains['p']['ki']
        self.p_kd = control_gains['p']['kd']

        self.q_kp = control_gains['q']['kp']
        self.q_ki = control_gains['q']['ki']
        self.q_kd = control_gains['q']['kd']

        self.r_kp = control_gains['r']['kp']
        self.r_ki = control_gains['r']['ki']
        self.r_kd = control_gains['r']['kd']

        self.h_kp = control_gains['h']['kp']
        self.w_kp = control_gains['w']['kp']
        self.w_ki = control_gains['w']['ki']

        self.dt = dt

        self.roll_ctrl = PIDController(self.roll_kp, 0., 0., self.dt, [-10, 10], [0,0])
        self.pitch_ctrl = PIDController(self.pitch_kp, 0., 0., self.dt, [-10, 10], [0,0])
        self.yaw_ctrl = PIDController(self.yaw_kp, 0., 0., self.dt, [-180, 180], [0,0], err_func=self._wrap_to_pi)

        self.p_ctrl = PIDController(self.p_kp, self.p_ki, self.p_kd, self.dt, [-4, 4], [-2,2])
        self.q_ctrl = PIDController(self.q_kp, self.q_ki, self.q_kd, self.dt, [-4, 4], [-2,2])
        self.r_ctrl = PIDController(self.r_kp, self.r_ki, self.r_kd, self.dt, [-4, 4], [-2,2])

        self.alt_ctrl = PIDController(self.h_kp, 0., 0., self.dt, [0, 500], [0,0])
        self.alt_rate_ctrl = PIDController(self.w_kp, self.w_ki, 0., self.dt)

    def _wrap_to_pi(self, psi_cmd, psi):
        error = psi_cmd - psi
        return ((error + np.pi) % (2*np.pi)) - np.pi

    def compute_control(self, setpoints:dict, measurements:dict):
        # Roll channel: ip -> phi_measured; op -> roll moment cmd, L_cmd
        phi_m = measurements['phi']
        p_m = measurements['p'] 
        phi_cmd = setpoints['phi']

        roll_loop = self.roll_ctrl.pid(phi_cmd, phi_m)
        L_cmd = self.p_ctrl.pid(roll_loop, p_m) 

        # pitch channel: ip -> theta_measured; op -> pitch moment cmd, M_cmd
        theta_m = measurements['theta']
        q_m = measurements['q'] 
        theta_cmd = setpoints['theta']

        pitch_loop = self.pitch_ctrl.pid(theta_cmd, theta_m)
        M_cmd = self.p_ctrl.pid(pitch_loop, q_m) 

        # yaw channel: ip -> psi_measured; op -> yaw moment cmd, N_cmd
        psi_m = measurements['psi']
        r_m = measurements['r']
        psi_cmd = setpoints['psi']

        psi_loop = self.yaw_ctrl.pid(psi_cmd, psi_m)
        N_cmd = self.r_ctrl.pid(psi_loop, r_m)

        # collective thrust/altitude channel: ip -> altitude (-z_n in metres); op -> collective thrust command, T_cmd
        h_m = measurements['h']
        vertical_velo_ned = measurements['w_ned']
        h_cmd = setpoints['h']
        # hover_thrust = setpoints['hover_thrust']

        alt_loop = self.alt_ctrl.pid(h_cmd, h_m)
        delta_T = self.alt_rate_ctrl.pid(alt_loop, vertical_velo_ned)
        delta_T *= 4
        # Tc_cmd = hover_thrust + delta_T

        return [delta_T, L_cmd, M_cmd, N_cmd]


