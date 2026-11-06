import numpy as np
import ussa1976
from collections import defaultdict
import os
import sys
import matplotlib.pyplot as plt
import math

# (Adjusting the number of os.path.dirname calls to find simulations submodule)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# 2. Add it to Python's search path if it's not already there
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from controller.multirotor.DJI_Mavic_Pro.pid_dji_mavic import CascadedPIDControl
from interface.multirotor.DJI_Mavic_Pro.vehicle_interface import VehicleInterface
from simulations.Aerospace.sixDOF.governing_equations.flat_earth_eom import flat_earth_eom
from simulations.Aerospace.sixDOF.numerical_integrators.numerical_integrator import rk4,rk4_timeseries
from simulations.Aerospace.sixDOF.vehicle_models.multirotor import quadcopter
from simulations.Aerospace.sixDOF.tools.Interpolators import interp

def main():
    dt_plant = 0.01
    dt_ctrl = 0.01
    sim_time = 10.

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
    u0 = vmod.u_trim
    nu0 = u0.size

    x = np.empty((nx0, n_t_plant), dtype=float)
    x[:, 0] = x0

    # timeseries control input
    time_data = np.array([0., 1., 1.1, 3., 3.1, 5., 5.1, 8., 8.1, 10.])
    delta = 5/57.3
    phi_cmd = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    theta_cmd = np.array([0., 0., -delta, -delta, 0., 0., delta, delta, 0., 0.])
    psi_cmd = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    alt_cmd = 200 * np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

    measurements = defaultdict(dict)
    setpoints = defaultdict(dict)
    vmod.rho_kgpm3 = 1.2
    trim_thrust = [vmod._thrust_model(u0[0])] * 4

    for i, t in enumerate(t_plant, start=1):
        print(t)
        setpoints['phi'] = interp(time_data, phi_cmd, t)
        setpoints['theta'] = interp(time_data, theta_cmd, t)
        setpoints['psi'] = interp(time_data, psi_cmd, t)
        setpoints['h'] = interp(time_data, alt_cmd, t)

        # x = u v w p q r phi theta psi x y h
        measurements['phi']   = x[6,i-1]
        measurements['p']     = x[3,i-1] 
        measurements['theta'] = x[7,i-1]
        measurements['q']     = x[4,i-1] 
        measurements['psi']   = x[8,i-1]
        measurements['r']     = x[5,i-1]
        measurements['h']     = -x[11,i-1]
        measurements['w_ned'] = x[2,i-1]

        rho = interp(amod['alt_m'], amod['rho_kgpm3'], -x[11,i-1])

        control_op = controller.compute_control(setpoints, measurements)

        rpm = vehicle_intf.u_cmd_perturb(control_op, trim_thrust, rho)

        x[:,i] , _ = rk4(flat_earth_eom, t, x[:,i-1], rpm, dt_plant, vmod=vmod, amod=amod)

if __name__ == '__main__':
    main()
