"""
    Generic PID controller parent class
"""

class PIDController:
    def __init__(self, Kp, Ki, Kd, output_limit, integral_limit, dt) -> None:
        self.kp = Kp
        self.ki = Ki
        self.kd = Kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.integral_state = 0.
        self.prev_error = 0.
        self.dt = dt

    def _update(self, setpoint, measurement):
        self.error = setpoint - measurement

    def _proportional(self):
        return self.kp * self.error

    def _integral(self):
        self.integral_state += self.error * self.dt
        return self.ki * self.integral_state

    def _derivative(self):
        self.prev_error = self.error
        return self.kd * (self.error - self.prev_error) / self.dt

    def pid(self, setpoint, measurement):
        self._update(setpoint, measurement)
        return self._proportional() + self._integral() + self._derivative()
