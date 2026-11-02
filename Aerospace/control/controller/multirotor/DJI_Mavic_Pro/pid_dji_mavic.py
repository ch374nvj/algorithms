from ...pid_control import PIDController

class CascadedPIDControl:
    def __init__(self, attitude_gains, rate_gains, dt) -> None:
        """Inits the PID controller object with control gains

        Args:
            attitude_gains (dict): contains roll, pitch and yaw P gains, packaged in a dict
            rate_gains (dict): contains p,q,r PID gains packaged in a dictionary
            dt (float): sample time
        """ 
        self.phi_kp = attitude_gains['roll_kp']
        self.theta_kp = attitude_gains['pitch_kp']
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
