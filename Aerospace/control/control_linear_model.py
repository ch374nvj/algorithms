import numpy as np
import pandas as pd
import seaborn as sns
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import sys
from interface.frame_transformation import FrameTransformer

# Conflicting Env variables fix
# 1. Force Matplotlib to use the non-interactive backend everywhere
os.environ["MPLBACKEND"] = "Agg"

# 2. Clean up conflicting Snap/system environment variables before libraries load
for var in ["LD_LIBRARY_PATH", "LD_PRELOAD", "GTK_PATH", "GIO_MODULE_DIR"]:
    if var in os.environ:
        del os.environ[var]

# 3. Re-execute the script if we need to completely purge the OS-level linker cache
# This acts exactly like running your `env -u` bash command automatically
if any(v in os.environ for v in ["LD_LIBRARY_PATH", "LD_PRELOAD"]):
    os.execv(sys.executable, [sys.executable] + sys.argv)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from controller.pid_control import PIDController
from interface.multirotor.DJI_Mavic_Pro.vehicle_interface import VehicleInterface
from tools.plotter import plot_results
from simulations.Aerospace.sixDOF.numerical_integrators.numerical_integrator import rk4
from simulations.Aerospace.sixDOF.tools.state_space import StateSpace
from simulations.Aerospace.sixDOF.vehicle_models.multirotor.quadcopter import Quad_X_DJI_Mavic_Pro

def pitch_channel_simple_rigid_model():
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
        theta_cmd = np.deg2rad(15.0) if t >= 1. and t < 3 else np.deg2rad(-10.0) if t >= 3  else 0.

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

def pitch_channel_state_space():
    dt = 0.01
    vmod = Quad_X_DJI_Mavic_Pro()
    v_inf = VehicleInterface()
    with np.load('../../simulations/Aerospace/sixDOF/dataset/DJI_mavic_pro_lon.npz') as data:
        lin_plant_A = data['Alon']
        lin_plant_B = data['Blon']

    lin_plant = StateSpace(lin_plant_A, lin_plant_B, np.eye(4), np.zeros((4,4)), dt)

    # Taking gains that were used in PID control of simple pitch model
    # Gains are tuned to minimize chattering, not fully gone though
    kp_q = 0.0432*0.2
    ki_q = 0.192*1.8
    kp_theta = 2

    q_ctrlr = PIDController(kp_q, ki_q, 0., dt, output_limit=[-50, 50], integral_limit=[0,0])
    theta_ctrlr  = PIDController(kp_theta, 0., 0., dt, output_limit=[-10,10])

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
    x = np.array([0.] * 4)
    rpm_u1 = []
    rpm_u2 = []
    rpm_u3 = []
    rpm_u4 = []
    

    for i, t in enumerate(t_s):

        # Pitch reference (rad)
        theta_cmd = np.deg2rad(8.0) if t > 1. else 0.

        # Pitch-rate reference (rad/s)
        # q_cmd = np.deg2rad(20.0) if t >= 0.5 else 0.0

        # P controller: commanded pitch
        q_cmd = theta_ctrlr.pid(theta_cmd, theta)
        theta_error = theta_ctrlr.error
        # PI controller: commanded pitch moment
        My_cmd = q_ctrlr.pid(q_cmd, q)
        q_error = q_ctrlr.error

        u = v_inf.u_cmd_perturb([0., 0., My_cmd, 0.], [0]*4, 1.2)

        x += dt * lin_plant._derivatives(0, x, u)
        q = x[2]
        theta = x[3]

        # Log signals
        err_history.append(theta_error)
        theta_history.append(theta)
        theta_cmd_history.append(theta_cmd)
        q_history.append(q)
        q_cmd_history.append(q_cmd)
        moment_history.append(My_cmd)
        rpm_u1.append(u[0])
        rpm_u2.append(u[1])
        rpm_u3.append(u[2])
        rpm_u4.append(u[3])

    # ####################################
    # Plotting using seaborn and pandas
    # ####################################
    # 1. Define your data lists in a dictionary
    df_sim = pd.DataFrame({
        "t_s": t_s,
        "theta": np.rad2deg(theta_history),
        "theta_cmd": np.rad2deg(theta_cmd_history),
        "q": q_history,
        "q_cmd": q_cmd_history,
        "moment_cmd": moment_history,
        "rpm1": rpm_u1,
        "rpm2": rpm_u2,
        "rpm3": rpm_u3,
        "rpm4": rpm_u4,
    })

    subplot_configs = [
        ["theta_cmd", "theta"],  # <-- Overlaid on the first subplot
        ["q_cmd", "q"],
        "moment_cmd",
    ]
    save_path = "savefig/linear/pitch_control_SS_step.png"

    plot_results(df_sim, subplot_configs, save_path)

def roll_channel_simple_rigid_model():
    dt = 0.01
    vmod = Quad_X_DJI_Mavic_Pro()
    v_inf = VehicleInterface()
    Ixx = vmod.Jxx_b_kgm2

    # Controller requirements,
    # pitch rate controller (PI):
    # choosing natural freq: 8 rad/s
    # damping ratio: 0.9
    # Analytically calculated gains,
    # kp = 2 * zeta * omega * Ixx = 0.0432
    # ki = omega^2 * Iyy = 0.192
    # Took same gains for pitch coz drone is symmetric 
    kp = 0.0432
    ki = 0.192

    p_ctrlr = PIDController(kp, ki, 0., dt, output_limit=[-50,50], integral_limit=[0,0])

    # pitch controller (P):
    # since the outer loop must be slower than inner loop, 
    # we choose kp < inner loop's natural freq
    #   kp < omega => kp < 8 rad/s
    kp_phi = 2
    phi_ctrlr = PIDController(kp_phi, 0., 0., dt, output_limit=[-10,10])

    # Set time conditions
    t0_s = 0.0
    tf_s = 5.
    h_s  = 0.01
    t_s = np.arange(t0_s, tf_s + h_s, h_s)
    nt_s = t_s.size

    # init conditions
    p = 0.0
    phi = 0.

    p_history = []
    p_cmd_history = []
    moment_history = []
    p_dot_history = []
    err_history = []
    phi_history = []
    phi_cmd_history = []

    for t in t_s:

        # Pitch reference (rad)
        phi_cmd = np.deg2rad(15.0) if t >= 1. and t < 3 else np.deg2rad(-10.0) if t >= 3  else 0.

        # Pitch-rate reference (rad/s)
        # q_cmd = np.deg2rad(20.0) if t >= 0.5 else 0.0

        # P controller: commanded roll
        p_cmd = phi_ctrlr.pid(phi_cmd, phi)
        phi_error = phi_ctrlr.error
        # PI controller: commanded pitch moment
        Mx_cmd = p_ctrlr.pid(p_cmd, p)
        p_error = p_ctrlr.error

        # Rigid-body dynamics
        p_dot = Mx_cmd / Ixx
        phi_dot = p

        # Integrate plant using Euler integration
        p += p_dot * dt
        phi += p * dt

        # Log signals
        err_history.append(phi_error)
        phi_history.append(phi)
        phi_cmd_history.append(phi_cmd)
        p_history.append(p)
        p_dot_history.append(p_dot)
        p_cmd_history.append(p_cmd)
        moment_history.append(Mx_cmd)

    df_sim = pd.DataFrame({
        "t_s": t_s,
        "phi": np.rad2deg(phi_history),
        "phi_cmd": np.rad2deg(phi_cmd_history),
        "p": p_history,
        "p_cmd": p_cmd_history,
        "moment_cmd": moment_history,
    })

    subplot_config = [
        ["phi_cmd", "phi"],
        ["p_cmd", "p"],
        ["moment_cmd"],
    ]

    plot_results(df_sim, subplot_config, 'savefig/linear/roll_control_TF.png')

def roll_channel_state_space():
    dt = 0.01
    vmod = Quad_X_DJI_Mavic_Pro()
    v_inf = VehicleInterface()
    with np.load('../../simulations/Aerospace/sixDOF/dataset/DJI_mavic_pro_lat.npz') as data:
        lin_plant_A = data['Alat']
        lin_plant_B = data['Blat']

    lin_plant = StateSpace(lin_plant_A, lin_plant_B, np.eye(5), np.zeros((5,4)), dt)

    # Taking gains that were used in PID control of simple pitch model
    # Gains are tuned to minimize chattering, not fully gone though
    kp_p = 0.0432 * 0.2
    ki_p = 0.192 * 1.8
    kp_phi = 2 *1 

    p_ctrlr = PIDController(kp_p, ki_p, 0., dt, output_limit=[-50, 50], integral_limit=[0,0])
    phi_ctrlr  = PIDController(kp_phi, 0., 0., dt, output_limit=[-10,10])

    # Set time conditions
    t0_s = 0.0
    tf_s = 5.
    h_s  = 0.01
    t_s = np.arange(t0_s, tf_s + h_s, h_s)
    nt_s = t_s.size

    # init conditions
    p = 0.0
    phi = 0.

    p_history = []
    p_cmd_history = []
    moment_history = []
    p_dot_history = []
    err_history = []
    phi_history = []
    phi_cmd_history = []
    x = np.array([0.] * 5)
    rpm_u1 = []
    rpm_u2 = []
    rpm_u3 = []
    rpm_u4 = []
    

    for i, t in enumerate(t_s):

        # roll reference (rad)
        phi_cmd = np.deg2rad(8.0) if t > 1. else 0.

        # P controller: commanded pitch
        p_cmd = phi_ctrlr.pid(phi_cmd, phi)
        phi_error = phi_ctrlr.error
        # PI controller: commanded pitch moment
        Mx_cmd = p_ctrlr.pid(p_cmd, p)
        p_error = p_ctrlr.error

        u = v_inf.u_cmd_perturb([0., Mx_cmd, 0., 0.], [0]*4, 1.2)

        x += dt * lin_plant._derivatives(0, x, u)
        p = x[1]
        phi = x[3]

        # Log signals
        err_history.append(phi_error)
        phi_history.append(phi)
        phi_cmd_history.append(phi_cmd)
        p_history.append(p)
        p_cmd_history.append(p_cmd)
        moment_history.append(Mx_cmd)
        rpm_u1.append(u[0])
        rpm_u2.append(u[1])
        rpm_u3.append(u[2])
        rpm_u4.append(u[3])

    df_sim = pd.DataFrame({
        "t_s": t_s,
        "phi": np.rad2deg(phi_history),
        "phi_cmd": np.rad2deg(phi_cmd_history),
        "p": p_history,
        "p_cmd": p_cmd_history,
        "moment_cmd": moment_history,
        "rpm1": rpm_u1,
        "rpm2": rpm_u2,
        "rpm3": rpm_u3,
        "rpm4": rpm_u4,
        })
    
    subplot_config = [
            ["phi_cmd", "phi"],
            ["p_cmd", "p"],
            ["moment_cmd"],
        ]
    
    plot_results(df_sim, subplot_config, 'savefig/linear/roll_control_step_SS.png')

def yaw_channel_simple_rigid_model():
    ...

def yaw_channel_state_space():
    dt = 0.01
    vmod = Quad_X_DJI_Mavic_Pro()
    v_inf = VehicleInterface()
    with np.load('../../simulations/Aerospace/sixDOF/dataset/DJI_mavic_pro_lat.npz') as data:
        lin_plant_A = data['Alat']
        lin_plant_B = data['Blat']

    lin_plant = StateSpace(lin_plant_A, lin_plant_B, np.eye(5), np.zeros((5,4)), dt)

    # Controller requirements,
    # pitch rate controller (PI):
    # choosing natural freq: 8 rad/s
    # damping ratio: 0.9
    # Analytically calculated gains,
    # kp = 2 * zeta * omega * Izz = 2*0.9*8*0.007 = 0.1008
    # ki = omega^2 * Izz = 8*8*0.007 = 0.448
    kp_r = 0.1008
    ki_r = 0.448

    r_ctrlr = PIDController(kp_r, ki_r, 0., dt, output_limit=[-50, 50], integral_limit=[0,0])
    # pitch controller (P):
    # since the outer loop must be slower than inner loop, 
    # we choose kp < inner loop's natural freq
    #   kp < omega => kp < 8 rad/s
    kp_psi = 2
    psi_ctrlr  = PIDController(kp_psi, 0., 0., dt, output_limit=[-10,10])

    # Set time conditions
    t0_s = 0.0
    tf_s = 5.
    h_s  = 0.01
    t_s = np.arange(t0_s, tf_s + h_s, h_s)
    nt_s = t_s.size

    # init conditions
    r = 0.0
    psi = 0.

    r_history = []
    r_cmd_history = []
    moment_history = []
    r_dot_history = []
    err_history = []
    psi_history = []
    psi_cmd_history = []
    x = np.array([0.] * 5)
    rpm_u1 = []
    rpm_u2 = []
    rpm_u3 = []
    rpm_u4 = []
    

    for i, t in enumerate(t_s):

        # roll reference (rad)
        psi_cmd = np.deg2rad(90.0) if t > 1. else 0.

        # P controller: commanded pitch
        r_cmd = psi_ctrlr.pid(psi_cmd, psi)
        psi_error = psi_ctrlr.error
        # PI controller: commanded pitch moment
        Mz_cmd = r_ctrlr.pid(r_cmd, r)
        r_error = r_ctrlr.error

        u = v_inf.u_cmd_perturb([0., 0., 0., Mz_cmd], [0]*4, 1.2)

        x += dt * lin_plant._derivatives(0, x, u)
        r = x[2]
        psi = x[4]

        # Log signals
        err_history.append(psi_error)
        psi_history.append(psi)
        psi_cmd_history.append(psi_cmd)
        r_history.append(r)
        r_cmd_history.append(r_cmd)
        moment_history.append(Mz_cmd)
        rpm_u1.append(u[0])
        rpm_u2.append(u[1])
        rpm_u3.append(u[2])
        rpm_u4.append(u[3])

    df_sim = pd.DataFrame({
        "t_s": t_s,
        "psi": np.rad2deg(psi_history),
        "psi_cmd": np.rad2deg(psi_cmd_history),
        "r": r_history,
        "r_cmd": r_cmd_history,
        "moment_cmd": moment_history,
        "rpm1": rpm_u1,
        "rpm2": rpm_u2,
        "rpm3": rpm_u3,
        "rpm4": rpm_u4,
        })
    
    subplot_config = [
            ["psi_cmd", "psi"],
            ["r_cmd", "r"],
            ["moment_cmd"],
        ]
    
    plot_results(df_sim, subplot_config, 'savefig/linear/yaw_control_step_SS.png')

def alt_channel_rigid_model():
    dt = 0.01
    vmod = Quad_X_DJI_Mavic_Pro()
    v_inf = VehicleInterface()
    mass = vmod.m_kg
    f_tfr = FrameTransformer()

    # Vertical velocity controller (PI)
    # Analytically computed,
    # kp = 2*zeta*omega*mass = 10.6992
    # ki = omega^2 * mass = 47.552
    v_kp = 10.6992
    v_ki = 47.552

    v_ctrlr = PIDController(v_kp, v_ki, 0., dt, output_limit=[-15,15])

    # Altitude controller (P)
    # analytically computed,
    # kp < omega_n, kp = 6
    alt_kp = 2
    alt_ctrlr = PIDController(alt_kp, 0., 0., dt, output_limit=[-10, 10])

    # Set time conditions
    t0_s = 0.0
    tf_s = 60.
    h_s  = 0.01
    t_s = np.arange(t0_s, tf_s + h_s, h_s)
    nt_s = t_s.size

    v = 0.
    alt = 10.

    alt_history = []
    alt_cmd_history = []
    v_history = []
    v_cmd_history = []
    err_history = []
    thrust_cmd_history = []

    for t in t_s:

        # Altitude reference (m)
        alt_cmd = 200. if t >= 1. else 0.
        # Vertical velocity controller
        v_cmd = alt_ctrlr.pid(alt_cmd, alt)
        # collective thrust controller (delta T)
        thrust_cmd = v_ctrlr.pid(v_cmd, v)

        # rigid body vertical velocity dynamics
        vdot = thrust_cmd/mass + 9.81
        v += vdot * dt
        alt += v * dt

        alt_cmd_history.append(alt_cmd)
        alt_history.append(alt)
        v_cmd_history.append(v_cmd)
        v_history.append(v)
        thrust_cmd_history.append(thrust_cmd)

    # plotting

    df_sim = pd.DataFrame({
        "t_s": t_s,
        "alt": alt_history,
        "alt_cmd": alt_cmd_history,
        "v": v_history,
        "v_cmd": v_cmd_history,
        "collec_thrust_cmd": thrust_cmd_history
    })

    subplot_configs = [
        ["alt", "alt_cmd"],
        ["v", "v_cmd"],
        "collec_thrust_cmd",
    ]

    save_path = "savefig/linear/alt_ctrl_rigid_step_with_thrust_constraint.png"

    plot_results(df_sim, subplot_configs, save_path)

def alt_channel_state_space():
    dt = 0.01
    vmod = Quad_X_DJI_Mavic_Pro()
    v_inf = VehicleInterface()
    f_tfr = FrameTransformer()

    with np.load('../../simulations/Aerospace/sixDOF/dataset/DJI_mavic_pro_lon.npz') as data:
        lin_plant_A = data['Alon']
        lin_plant_B = data['Blon']

    lin_plant = StateSpace(lin_plant_A, lin_plant_B, np.eye(4), np.zeros((4,4)), dt)
    # print(lin_plant)
    # Controller requirements,
    # pitch rate controller (PI):
    # choosing natural freq: 8 rad/s
    # damping ratio: 0.9
    # Analytically calculated gains,
    # kp = 2*zeta*omega*mass = 10.6992
    # ki = omega^2 * mass = 47.552
    v_kp = 10.6992
    v_ki = 47.552

    v_ctrlr = PIDController(v_kp, v_ki, 0., dt, output_limit=[-2,2], integral_limit=[-1, 1])
    
    # Altitude controller (P)
    # analytically computed,
    # kp < omega_n
    alt_kp = 1
    alt_ctrlr = PIDController(alt_kp, 0., 0., dt, output_limit=[-5, 5], integral_limit=[-1,1])

    # Set time conditions
    t0_s = 0.0
    tf_s = 60.
    h_s  = 0.01
    t_s = np.arange(t0_s, tf_s + h_s, h_s)
    nt_s = t_s.size

    v = 0.
    alt = 0.

    alt_history = []
    alt_cmd_history = []
    v_history = []
    v_cmd_history = []
    err_history = []
    thrust_cmd_history = []
    x = np.array([0.] * 4)
    rpm_u1 = []
    rpm_u2 = []
    rpm_u3 = []
    rpm_u4 = []
    hover_thrust = np.array([vmod.m_kg * 9.81] * 4) # hover thrust = 0.743 * 9.81 ~= 7.3N
    hover_rpm = np.array([11197.]*4) # at 11197 RPM thrust is ~7.3N

    for t in t_s:
        # Altitude reference (m)
        alt_cmd = 200. if t >= 1. else 0.
        # Vertical velocity controller
        v_cmd = alt_ctrlr.pid(alt_cmd, alt)
        err_history.append(alt_ctrlr.error)
        # collective thrust controller (delta T)
        delta_thrust_cmd = v_ctrlr.pid(v_cmd, v)

        u = v_inf.u_cmd_perturb([delta_thrust_cmd, 0., 0., 0.], hover_thrust, 1.2)
        u -= hover_rpm
        x += dt * lin_plant._derivatives(0, x, u)
        # v = -x[1] # v_climb ~ (-w) 
        v = f_tfr.vertical_velo_ned(
            x[0],
            0.,
            x[1],
            0.,
            x[3],
            0.,
        ) # transforming body frame velo to NED
        alt += v * dt

        # Log signals
        alt_cmd_history.append(alt_cmd)
        alt_history.append(alt)
        v_cmd_history.append(v_cmd)
        v_history.append(v)
        thrust_cmd_history.append(delta_thrust_cmd)
        rpm_u1.append(u[0])
        rpm_u2.append(u[1])
        rpm_u3.append(u[2])
        rpm_u4.append(u[3])

    # plotting
    
    df_sim = pd.DataFrame({
        "t_s": t_s,
        "err_history":err_history,
        "alt": alt_history,
        "alt_cmd": alt_cmd_history,
        "v": v_history,
        "v_cmd": v_cmd_history,
        "collec_thrust_cmd": thrust_cmd_history,
        "delta_rpm_u1":rpm_u1,
        "delta_rpm_u2":rpm_u2,
        "delta_rpm_u3":rpm_u3,
        "delta_rpm_u4":rpm_u4,
    })

    subplot_configs = [
        ["alt", "alt_cmd"],
        ["v", "v_cmd"],
        "collec_thrust_cmd",
        ["delta_rpm_u1", "delta_rpm_u2"],
        "err_history"
    ]

    save_path = "savefig/linear/alt_ctrl_state_space_with_constraint_2.png"

    plot_results(df_sim, subplot_configs, save_path, ncols=2)



if __name__ == '__main__':
    # pitch_channel_simple_rigid_model()
    # pitch_channel_state_space()
    # roll_channel_simple_rigid_model()
    # roll_channel_state_space()
    # yaw_channel_simple_rigid_model()
    # yaw_channel_state_space()
    # alt_channel_rigid_model()
    alt_channel_state_space()
    ...