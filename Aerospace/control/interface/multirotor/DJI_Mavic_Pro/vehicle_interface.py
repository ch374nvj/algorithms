import numpy as np
from .actuator_model import ActuatorModel
from .ctrl_allocator import LocalAllocator

class VehicleInterface:
    def __init__(self, trim_rpm=5600, rho_trim=1.2) -> None:
        self._act_mod = ActuatorModel()
        kq = self._act_mod.drag_torque_coeff(rpm=trim_rpm, rho=rho_trim)
        self._alloc = LocalAllocator(kq)

    def u_cmd_perturb(self, u_vec, trim_thrust, rho):
        [Tc, L, M, N] = u_vec
        T_cmd = self._alloc.allocate(trim_thrust, u_vec)

        RPM_cmd = []
        for T in T_cmd:
            RPM_i = self._act_mod.rpm_from_thrust(T, rho)
            RPM_cmd.append(RPM_i)

        return np.asarray(RPM_cmd, dtype=float)