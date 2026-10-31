import numpy as np

class ActuatorModel:
    def __init__(self) -> None:

        self.CT_RPM_coeffs = np.array([2e-15, -4e-11, 3e-7, 0.1013])
        self.Q_RPM_coeffs = np.array([4e-14, 8e-12, 3e-6, 0])
        self.rpm_min = 0
        self.rpm_max = 16000
        self.prop_D_m = 0.2032

        self._inv_tolerance = 1e-4
        self._inv_max_iter = 100
        self._kq_perturb = 1e-4

    def thrust_from_rpm(self, rpm, rho):
        CT = np.sum(self.CT_RPM_coeffs * [rpm**3, rpm**2, rpm, 1])
        rps = rpm/60
        thrust = CT * rho * (rps**2) * (self.prop_D_m**4)
        return thrust

    def torque_from_rpm(self,rpm):
        Q = np.sum(self.Q_RPM_coeffs * [rpm**3, rpm**2, rpm, 1])
        return Q

    def rpm_from_thrust(self, thrust, rho):
        # Numerically inverting thrust model using bisection

        thrust_min = self.thrust_from_rpm(self.rpm_min, rho)
        thrust_max = self.thrust_from_rpm(self.rpm_max, rho)

        if not thrust_min <= thrust <= thrust_max:
            raise ValueError("Requested thrust is outside actuator limits.")

        low_rpm = float(self.rpm_min)
        high_rpm = float(self.rpm_max)

        for _ in range(self._inv_max_iter):
            mid_rpm = 0.5 * (low_rpm + high_rpm)
            thrust_mid = self.thrust_from_rpm(mid_rpm, rho)

            if abs(thrust - thrust_mid) < self._inv_tolerance:
                return int(round(mid_rpm))

            if thrust < thrust_mid:
                high_rpm = mid_rpm
            else:
                low_rpm = mid_rpm

        return int(round(0.5 * (high_rpm + low_rpm)))

    def drag_torque_coeff(self, rpm, rho):
        kq = (self.torque_from_rpm(rpm +self._kq_perturb) - self.torque_from_rpm(rpm-self._kq_perturb))/(self.thrust_from_rpm(rpm +self._kq_perturb, rho) - self.thrust_from_rpm(rpm -self._kq_perturb, rho))
        return kq
