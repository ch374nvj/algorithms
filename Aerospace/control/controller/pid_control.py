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
        if output_limit == [0,0]:
            self.u_limit_en = False
        else:
            self.u_limit_en = True
            self.out_min, self.out_max = output_limit
        if integral_limit == [0,0]:
            self.i_limit_en = False
        else:
            self.i_limit_en = True 
            self.int_min, self.int_max = integral_limit

    def _error(self, setpoint,measurement):
        return setpoint - measurement
    
    def _update(self, setpoint, measurement):
        self.error = self.err_func(setpoint, measurement) # type: ignore (for pylance)

    def update_gains(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def reset(self):
        self.integral_state = 0.
        self.prev_error = 0.

    def _proportional(self):
        return self.kp * self.error
    
    def _integral(self):
        integral_np1 = self.integral_state + self.ki * self.error * self.dt
        if self.i_limit_en:
            # Conditional integration for anti-windup
            integral_np1 = max(self.int_min, min(integral_np1, self.int_max))
            if (self.u in self.output_limit or (self.u > self.out_max and self.error < 0) 
                or (self.u < self.out_min and self.error > 0)):
                self.integral_state = integral_np1
            else:
                self.integral_state += integral_np1
        else:
            self.integral_state += integral_np1
        return self.integral_state

    def _derivative(self):
        self.prev_error = self.error
        return self.kd * (self.error - self.prev_error) / self.dt

    def pid(self, setpoint, measurement):
        self._update(setpoint, measurement)
        self.u = self._proportional() + self._integral() + self._derivative()
        if self.u_limit_en:
            self.u = max(self.out_min, min(self.u, self.out_max))
        return self.u

    @property
    def P(self):
        return self._proportional()

    @property
    def I(self):
        return self._integral()

    @property
    def D(self):
        return self._derivative()

