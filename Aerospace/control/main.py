import numpy as np
import ussa1976
from collections import defaultdict
import os
import sys
# import matplotlib.pyplot as plt
import pandas as pd
import math

# (Adjusting the number of os.path.dirname calls to find simulations submodule)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# 2. Add it to Python's search path if it's not already there
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from controller.multirotor.DJI_Mavic_Pro.pid_dji_mavic import CascadedPIDControl
from interface.multirotor.DJI_Mavic_Pro.vehicle_interface import VehicleInterface
from tools.plotter import plot_results
from simulations.Aerospace.sixDOF.governing_equations.flat_earth_eom import flat_earth_eom
from simulations.Aerospace.sixDOF.numerical_integrators.numerical_integrator import rk4,rk4_timeseries
from simulations.Aerospace.sixDOF.vehicle_models.multirotor import quadcopter
from simulations.Aerospace.sixDOF.tools.Interpolators import interp
from simulations.Aerospace.sixDOF.tools.state_space import StateSpace

def main():
    dt_plant = 0.01
    dt_ctrl = 0.01
    sim_time = 10.

    t0 = 0

    t_plant = np.arange(t0, sim_time + dt_plant, dt_plant)
    t_ctrl = np.arange(t0, sim_time + dt_ctrl, dt_ctrl)
    n_t_plant = t_plant.size
    n_t_ctrl = t_ctrl.size

    np.set_printoptions(precision=4, suppress=True)

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

    control_gains['roll']['kp'] = 2
    control_gains['pitch']['kp']= 2.5
    control_gains['yaw']['kp']  = 2

    control_gains['p']['kp']    = 0.00864 
    control_gains['p']['ki']    = 0.3456 * 1
    control_gains['p']['kd']    = 0.

    control_gains['q']['kp']    = 0.00864 * 2
    control_gains['q']['ki']    = 0.3456 * 2
    control_gains['q']['kd']    = 0.

    control_gains['r']['kp']    = 0.1008
    control_gains['r']['ki']    = 0.448
    control_gains['r']['kd']    = 0.

    control_gains['h']['kp']    = 0.
    control_gains['w']['kp']    = 0.
    control_gains['w']['ki']    = 0.

    vmod = quadcopter.Quad_X_DJI_Mavic_Pro()
    vehicle_intf = VehicleInterface()
    controller = CascadedPIDControl(control_gains, dt_ctrl)

    # Linear plant initialization
    with np.load('../../simulations/Aerospace/sixDOF/dataset/DJI_mavic_pro_AB.npz') as data:
        lin_plant_A = data['A']
        lin_plant_B = data['B']

    lin_plant = StateSpace(lin_plant_A, lin_plant_B, np.eye(12), np.zeros((12,4)), dt_plant)
    
    # initial states and trim ctrl input settings
    # also preallocating data arrays for non-realtime sim

    x0 = vmod.x_trim
    nx0 = x0.size
    u0 = vmod.u_trim
    nu0 = u0.size

    # State array
    x = np.empty((nx0, n_t_plant), dtype=float)
    # derivatives array
    dx = np.empty((nx0, n_t_plant), dtype=float)
    # attitude and altitude command array
    v_cmd = np.empty((4, n_t_ctrl), dtype=float) 
    # moment and collective thrust commands array
    v = np.empty((4, n_t_ctrl), dtype=float)
    # rpm commands array
    u = np.empty((nu0, n_t_ctrl), dtype=float)

    simulation_test = 'quad_pi_tuned_ctrl_test_roll_doublet_psi_cmd_nl'
    test_count = 2
    # setting trim states as init value for sim
    x[:, 0] = x0

    # timeseries control input
    time_data = np.array([0., 1., 1.1, 3., 3.1, 5., 5.1, 8., 8.1, 10.])
    delta = 5/57.3
    phi_cmd = np.array([0., 0., -delta, -delta, 0., 0., delta, delta, 0., 0.])
    # theta_cmd = -1 * np.array([0., 0., -delta, -delta, 0., 0., delta, delta, 0., 0.])
    theta_cmd = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    # theta_cmd = np.array([0.] * 10)
    psi_cmd = np.deg2rad(225) * np.array([1.] * 10)
    alt_cmd = 200 * np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

    measurements = defaultdict(dict)
    setpoints = defaultdict(dict)
    vmod.rho_kgpm3 = 1.2
    # trim_thrust = [vmod._thrust_model(u0[0])] * 4
    trim_thrust = [1.884] * 4

    for i, t in enumerate(t_plant[:-1], start=1):
        # print(t)

        v_cmd[0,i-1] = interp(time_data, phi_cmd, t)
        v_cmd[1,i-1] = interp(time_data, theta_cmd, t)
        v_cmd[2,i-1] = interp(time_data, psi_cmd, t)
        v_cmd[3,i-1] = interp(time_data, alt_cmd, t)

        [setpoints['phi'],
        setpoints['theta'],
        setpoints['psi'], 
        setpoints['h']] = v_cmd[:,i-1]

        # x = u v w p q r phi theta psi x y h
        measurements['phi']   = x[6,i-1]
        measurements['p']     = x[3,i-1] 
        measurements['theta'] = x[7,i-1]
        measurements['q']     = x[4,i-1] 
        measurements['psi']   = x[8,i-1]
        measurements['r']     = x[5,i-1]
        measurements['h']     = -x[11,i-1]
        measurements['w_ned'] = x[2,i-1]

        rho = vmod.rho_kgpm3

        v[:,i] = controller.compute_control(setpoints, measurements)
        # print(v[:,i]) # for debug
        u[:,i] = vehicle_intf.u_cmd_perturb(v[:,i], trim_thrust, rho)
        # print(u[:,i]) # for debug

        # Non-linear plant
        x[:,i] , _ = rk4(flat_earth_eom, t, x[:,i-1], u[:,i], dt_plant, vmod=vmod, amod=amod) 
        # print(x[3:9,i])
        # Linear plant
        # x[:, i] = lin_plant.outputs(x[:,i-1], u[:,i])
        # x[:, i] = lin_plant.step(u[:,i-1])
        # x[:, i] = x[:, i-1] + dt_plant * lin_plant._derivatives(0., x[:, i-1], u[:,i-1])

    # plotting

    df_sim = pd.DataFrame({
            "t_s": t_plant,
            "phi": np.rad2deg(x[6,:]),
            "phi_cmd": np.rad2deg(v_cmd[0,:]),
            "theta": np.rad2deg(x[7,:]),
            "theta_cmd": np.rad2deg(v_cmd[1,:]),
            "psi" : np.rad2deg(x[8,:]),
            "psi_cmd": np.rad2deg(v_cmd[2,:]),
            "alt": -x[11,:],
            "alt_cmd": v_cmd[3,:],
            "rpm1": u[0,:],
            "rpm2": u[1,:],
            "rpm3": u[2,:],
            "rpm4": u[3,:],
        })
        
    subplot_config = [
            ["phi_cmd", "phi"],
            ["theta_cmd", "theta"],
            ["psi_cmd", "psi"],
            ["alt_cmd", "alt"],
        ]
    
    plot_results(df_sim, subplot_config, f'savefig/{simulation_test}_{test_count}.png', ncols=4)

if __name__ == '__main__':
    main()
