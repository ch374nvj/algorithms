import numpy as np
import ussa1976
from collections import defaultdict
import os
import sys

# (Adjusting the number of os.path.dirname calls to find simulations submodule)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# 2. Add it to Python's search path if it's not already there
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from controller.multirotor.DJI_Mavic_Pro.pid_dji_mavic import CascadedPIDControl
from interface.multirotor.DJI_Mavic_Pro.vehicle_interface import VehicleInterface
from simulations.Aerospace.sixDOF.governing_equations import flat_earth_eom
from simulations.Aerospace.sixDOF.numerical_integrators.numerical_integrator import rk4,rk4_timeseries
from simulations.Aerospace.sixDOF.vehicle_models.multirotor import quadcopter

def main():
    dt_plant = 0.01
    dt_ctrl = 0.01
    sim_time = 20.

    t0 = 0

    t_plant = np.arange(t0, sim_time + dt_plant, dt_plant)
    t_ctrl = np.arange(t0, sim_time + dt_ctrl, dt_ctrl)
    n_t_plant = t_plant.size
    n_t_ctrl = t_ctrl.size

    # ####################################################
    #   plant, interface, controller & env initialization
    # ####################################################

    # Atmospheric data
    atmosphere = ussa1976.compute()

    # Essential gravity and atmospheric data
    amod_alt_m       = atmosphere['z'].values
    amod_rho_kgpm3   = atmosphere['rho'].values
    amod_c_mps       = atmosphere['cs'].values
    amod_g_mps2      = ussa1976.core.compute_gravity(amod_alt_m)

    # amod: atmospheric model with gravity
    amod = {
        "alt_m"     : amod_alt_m,
        "rho_kgpm3" : amod_rho_kgpm3,
        "c_mps"     : amod_c_mps,
        "g_mps2"    : amod_g_mps2
    }

    # Controller gains
    control_gains = defaultdict(dict)

    control_gains['roll']['kp'] = 0.1
    control_gains['pitch']['kp']= 0.1
    control_gains['yaw']['kp']  = 0.1

    control_gains['p']['kp']    = 0.1
    control_gains['p']['ki']    = 0.1
    control_gains['p']['kd']    = 0.1

    control_gains['q']['kp']    = 0.1
    control_gains['q']['ki']    = 0.1
    control_gains['q']['kd']    = 0.1

    control_gains['r']['kp']    = 0.1
    control_gains['r']['ki']    = 0.1
    control_gains['r']['kd']    = 0.1

    control_gains['h']['kp']    = 0.1
    control_gains['w']['kp']    = 0.1
    control_gains['w']['ki']    = 0.1

    vmod = quadcopter.Quad_X_DJI_Mavic_Pro()
    vehicle_intf = VehicleInterface()
    controller = CascadedPIDControl(control_gains, dt_ctrl)

    # initial states and trim ctrl input settings
    # also preallocating data arrays for non-realtime sim

    x0 = vmod.x_trim
    nx0 = x0.size

    x = np.empty((nx0, n_t_plant), dtype=float)

    for i, t in enumerate(t_plant):
        
        pass

if __name__ == '__main__':
    main()
