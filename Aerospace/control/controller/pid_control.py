"""
    Generic PID controller parent class
"""

class PIDController:
    def __init__(self, Kp, Ki, Kd, dt, output_limit = [-10,10], integral_limit = [-10,10], err_func=None) -> None:
        self.kp = Kp
        self.ki = Ki
        self.kd = Kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.integral_state = 0.
        self.prev_error = 0.
        self.dt = dt
        self.err_func = err_func  # provision to use a custom error function over a usual subtraction
        if err_func is None:
            self.err_func = self._error

    def _error(self, setpoint,measurement):
        return setpoint - measurement
    
    def _update(self, setpoint, measurement):
        self.error = self.err_func(setpoint, measurement) # type: ignore (for pylance)

    def reset(self):
        self.integral_state = 0.
        self.prev_error = 0.

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

    @property
    def P(self):
        return self._proportional()

    @property
    def I(self):
        return self._integral()

    @property
    def D(self):
        return self._derivative()

