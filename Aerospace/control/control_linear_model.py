import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from controller.pid_control import PIDController
from interface.multirotor.DJI_Mavic_Pro.vehicle_interface import VehicleInterface
# from simulations.Aerospace.sixDOF.tools.state_space import StateSpace
from simulations.Aerospace.sixDOF.vehicle_models.multirotor.quadcopter import Quad_X_DJI_Mavic_Pro

def pitch_channel():
    dt = 0.01
    vmod = Quad_X_DJI_Mavic_Pro()
    v_inf = VehicleInterface()
    Iyy = vmod.Jyy_b_kgm2

    # Controller requirements,
    # pitch rate controller (PI):
    # choosing natural freq: 8 rad/s
    # damping ratio: 0.9
    # Analytically calculated gains,
    # kp = 2 * zeta * omega * Iyy = 0.0432
    # ki = omega^2 * Iyy = 0.192
    kp = 0.0432
    ki = 0.192

    q_ctrlr = PIDController(kp, ki, 0., dt, output_limit=[-50,50], integral_limit=[0,0])

    # pitch controller (P):
    # since the outer loop must be slower than inner loop, 
    # we choose kp < inner loop's natural freq
    #   kp < omega => kp < 8 rad/s
    kp_theta = 2
    theta_ctrlr = PIDController(kp_theta, 0., 0., dt, output_limit=[-10,10])

    # Set time conditions
    t0_s = 0.0
    tf_s = 5.
    h_s  = 0.01
    t_s = np.arange(t0_s, tf_s + h_s, h_s)
    nt_s = t_s.size

    # init conditions
    q = 0.0
    theta = 0.

    q_history = []
    q_cmd_history = []
    moment_history = []
    q_dot_history = []
    err_history = []
    theta_history = []
    theta_cmd_history = []

    for t in t_s:

        # Pitch reference (rad)
        theta_cmd = np.deg2rad(15.0) if t >= 1. and t < 3 else np.deg2rad(-15.0) if t >= 3  else 0.

        # Pitch-rate reference (rad/s)
        # q_cmd = np.deg2rad(20.0) if t >= 0.5 else 0.0

        # P controller: commanded pitch
        q_cmd = theta_ctrlr.pid(theta_cmd, theta)
        theta_error = theta_ctrlr.error
        # PI controller: commanded pitch moment
        My_cmd = q_ctrlr.pid(q_cmd, q)
        q_error = q_ctrlr.error

        # Rigid-body dynamics
        q_dot = My_cmd / Iyy
        theta_dot = q

        # Integrate plant using Euler integration
        q += q_dot * dt
        theta += q * dt

        # Log signals
        err_history.append(theta_error)
        theta_history.append(theta)
        theta_cmd_history.append(theta_cmd)
        q_history.append(q)
        q_dot_history.append(q_dot)
        q_cmd_history.append(q_cmd)
        moment_history.append(My_cmd)


    # ==========================================
    # 5. Plot results
    # ==========================================

    fig, ax = plt.subplots(4, 1, figsize=(10, 7), sharex=True)

    ax[0].plot(t_s, np.rad2deg(theta_cmd_history), '--', label="Command")
    ax[0].plot(t_s, np.rad2deg(theta_history), label="Actual")
    ax[0].set_ylabel("Pitch (deg)")
    ax[0].set_title("Pitch P tracking")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(t_s, moment_history)
    ax[1].set_ylabel("Pitch moment (N.m)")
    ax[1].set_xlabel("Time (s)")
    # ax[1].set_title("Controller output")
    ax[1].grid(True)

    ax[2].plot(t_s, err_history)
    ax[2].set_ylabel("Error history")
    ax[2].set_xlabel("Time (s)")
    # ax[2].set_title("Controller output")
    ax[2].grid(True)

    ax[3].plot(t_s, q_history)
    ax[3].set_ylabel("q history")
    ax[3].set_xlabel("Time (s)")
    # ax[3].set_title("Controller output")
    ax[3].grid(True)


    plt.savefig(f'savefig/linear/pitch_rate_TF_control_doublet.png')

pitch_channel()