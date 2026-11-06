import numpy as np
from .actuator_model import ActuatorModel
from .ctrl_allocator import LocalAllocator

class VehicleInterface:
    def __init__(self) -> None:
        self.act_mod = ActuatorModel()
        kq = self.act_mod.drag_torque_coeff(rpm=5600, rho=1.2)
        self.alloc = LocalAllocator(kq)

    def u_cmd_perturb(self, u_vec, trim_thrust, rho):
        [Tc, L, M, N] = u_vec
        T_cmd = self.alloc.allocate(trim_thrust, u_vec)

        RPM_cmd = []
        for T in T_cmd:
            RPM_i = self.act_mod.rpm_from_thrust(T, rho)
            RPM_cmd.append(RPM_i)

        return RPM_cmd