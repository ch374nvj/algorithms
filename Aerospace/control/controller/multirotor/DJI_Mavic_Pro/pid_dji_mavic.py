from ...pid_control import PIDController
import numpy as np

class CascadedPIDControl:
    def __init__(self, control_gains: dict, dt:float) -> None:
        """Inits the PID controller object with control gains

        Args:
            control_gains (dict): contains all the control gains, packaged in a dict
            dt (float): sample time
        """ 
        self.roll_kp = control_gains.get('roll', {}).get('kp', 0.)
        self.pitch_kp = control_gains.get('pitch', {}).get('kp', 0.)
        self.yaw_kp = control_gains.get('yaw', {}).get('kp', 0.)

        self.p_kp = control_gains.get('p', {}).get('kp', 0.)
        self.p_ki = control_gains.get('p', {}).get('ki', 0.)
        self.p_kd = control_gains.get('p', {}).get('kd', 0.)

        self.q_kp = control_gains.get('q', {}).get('kp', 0.)
        self.q_ki = control_gains.get('q', {}).get('ki', 0.)
        self.q_kd = control_gains.get('q', {}).get('kd', 0.)

        self.r_kp = control_gains.get('r', {}).get('kp', 0.)
        self.r_ki = control_gains.get('r', {}).get('ki', 0.)
        self.r_kd = control_gains.get('r', {}).get('kd', 0.)

        self.h_kp = control_gains.get('h', {}).get('kp', 0.)
        self.w_kp = control_gains.get('w', {}).get('kp', 0.)
        self.w_ki = control_gains.get('w', {}).get('ki', 0.)
        self.w_kd = control_gains.get('w', {}).get('kd', 0.)

        self.dt = dt

        self.roll_ctrl = PIDController(self.roll_kp, 0., 0., self.dt, [-10, 10])
        self.pitch_ctrl = PIDController(self.pitch_kp, 0., 0., self.dt, [-10, 10])
        self.yaw_ctrl = PIDController(self.yaw_kp, 0., 0., self.dt, [-180, 180], err_func=self._wrap_to_pi)

        self.p_ctrl = PIDController(self.p_kp, self.p_ki, self.p_kd, self.dt, [-4, 4], [-2,2])
        self.q_ctrl = PIDController(self.q_kp, self.q_ki, self.q_kd, self.dt, [-4, 4], [-2,2])
        self.r_ctrl = PIDController(self.r_kp, self.r_ki, self.r_kd, self.dt, [-4, 4], [-2,2])

        self.alt_ctrl = PIDController(self.h_kp, 0., 0., self.dt, output_limit=[-5, 5], integral_limit=[-1,1])
        self.alt_rate_ctrl = PIDController(self.w_kp, self.w_ki, 0., self.dt, output_limit=[-2,2], integral_limit=[-1, 1])

    def _wrap_to_pi(self, psi_cmd, psi):
        error = psi_cmd - psi
        return ((error + np.pi) % (2*np.pi)) - np.pi

    def compute_control(self, setpoints:dict, measurements:dict):
        # Roll channel: ip -> phi_measured; op -> roll moment cmd, L_cmd
        phi_m = measurements.get('phi', 0.)
        p_m = measurements.get('p', 0.)
        phi_cmd = setpoints.get('phi', 0.)

        self.p_cmd = self.roll_ctrl.pid(phi_cmd, phi_m)
        self.L_cmd = self.p_ctrl.pid(self.p_cmd, p_m) 

        # pitch channel: ip -> theta_measured; op -> pitch moment cmd, M_cmd
        theta_m = measurements.get('theta', 0.)
        q_m = measurements.get('q', 0.)
        theta_cmd = setpoints.get('theta', 0.)

        self.q_cmd = self.pitch_ctrl.pid(theta_cmd, theta_m)
        self.M_cmd = self.p_ctrl.pid(self.q_cmd, q_m) 

        # yaw channel: ip -> psi_measured; op -> yaw moment cmd, N_cmd
        psi_m = measurements.get('psi', 0.)
        r_m = measurements.get('r', 0.)
        psi_cmd = setpoints.get('psi', 0.)

        self.r_cmd = self.yaw_ctrl.pid(psi_cmd, psi_m)
        self.N_cmd = self.r_ctrl.pid(self.r_cmd, r_m)

        # collective thrust/altitude channel: ip -> altitude (-z_n in metres); op -> collective thrust command, T_cmd
        h_m = measurements.get('h', 0.)
        vertical_velo_ned = measurements.get('w_ned', 0.)
        h_cmd = setpoints.get('h', 0.)
        # hover_thrust = setpoints['hover_thrust']

        self.v_cmd = self.alt_ctrl.pid(h_cmd, h_m)
        self.delta_T_cmd = self.alt_rate_ctrl.pid(self.v_cmd, vertical_velo_ned)
        # delta_T *= 4
        # Tc_cmd = hover_thrust + delta_T

        return [self.delta_T_cmd, self.L_cmd, self.M_cmd, self.N_cmd]


