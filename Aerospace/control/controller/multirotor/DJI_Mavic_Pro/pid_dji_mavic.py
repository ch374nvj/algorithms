from ...pid_control import PIDController

class CascadedPIDControl:
    def __init__(self, attitude_gains, rate_gains, dt) -> None:
        """Inits the PID controller object with control gains

        Args:
            attitude_gains (dict): contains roll, pitch and yaw P gains, packaged in a dict
            rate_gains (dict): contains p,q,r PID gains packaged in a dictionary
            dt (float): sample time
        """ 
        self.roll_kp = attitude_gains['roll_kp']
        self.pitch_kp = attitude_gains['pitch_kp']
        self.yaw_kp = attitude_gains['yaw_kp']

        self.p_kp = rate_gains['p']['kp']
        self.p_ki = rate_gains['p']['ki']
        self.p_kd = rate_gains['p']['kd']

        self.q_kp = rate_gains['q']['kp']
        self.q_ki = rate_gains['q']['ki']
        self.q_kd = rate_gains['q']['kd']

        self.r_kp = rate_gains['r']['kp']
        self.r_ki = rate_gains['r']['ki']
        self.r_kd = rate_gains['r']['kd']

        self.dt = dt

        self.roll_ctrl = PIDController(self.roll_kp, 0., 0., [-10, 10], [0,0], self.dt)
        self.pitch_ctrl = PIDController(self.pitch_kp, 0., 0., [-10, 10], [0,0], self.dt)

        self.p_ctrl = PIDController(self.p_kp, self.p_ki, self.p_kd, [-4, 4], [-2,2], self.dt)
        self.q_ctrl = PIDController(self.q_kp, self.q_ki, self.q_kd, [-4, 4], [-2,2], self.dt)
        self.r_ctrl = PIDController(self.r_kp, self.r_ki, self.r_kd, [-4, 4], [-2,2], self.dt)

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