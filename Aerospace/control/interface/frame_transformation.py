import numpy as np

class FrameTransformer:
    def __init__(self) -> None:
        ...

    def _update(self, u, v, w, phi, theta, psi):
        self.u = u
        self.v = v
        self.w = w
        self.phi = phi
        self.theta = theta
        self.psi = psi

    def vertical_velo_ned(self, u, v, w, phi, theta, psi):
        """
            For now, adding only vertical velocity translation from body to inertial (ned)
            for the controller
        """
        self._update(u, v, w, phi, theta, psi)
        return -1. * (-self.u * np.sin(self.theta) + self.v * np.sin(self.phi) * np.cos(self.theta) + self.w * np.cos(self.phi) * np.cos(self.theta))