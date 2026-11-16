# Quadcopter Control Algorithms

This repository contains control algorithms, simulations, and testing frameworks for a quadcopter platform. The system implements cascaded PID controllers for attitude and altitude control, with support for both nonlinear and linear plant models.


> [!TIP]
> The algorithms simulation code is entirely written in python and heavily uses classes and objects (My fav). If it is a little hard to follow the code, I recommend you to open the specific file or folder in github and click on the commit history on the right side (one with the clock symbol). That way you can see how the code was written chronologically.

## Overview

The control system is designed with a hierarchical structure:

- **High-level controllers**: Roll, pitch, yaw, and altitude angle control (outer loops)
- **Low-level controllers**: Roll rate, pitch rate, yaw rate, and vertical velocity control (inner loops)
- **Vehicle interface**: Actuator models and thrust-to-RPM conversion
- **State estimation**: Frame transformation and coordinate system conversions. A fully fledged estimator like kalman filter is in the works.
- **Simulation framework**: Full 6-DOF nonlinear dynamics integration

## AI usage declaration
At a time when the use of AI LLMs is prevalent and helps speed up the code-writing process, I would like to clarify that no AI tools were used to write the code or develop the libraries, unless explicitly mentioned in the inline comments or in the commits (like [commit a196888](https://github.com/ch374nvj/algorithms/commit/a1968883745274e2868dced2f1bd4348f2d3fcbb) where I used AI to create a plotting code (saved lotta time)).

As a personal principle, I generally avoid using AI tools for my personal projects.  However, I do use them as learning aids to explore and understand theoretical concepts, clarify doubts, and deepen my understanding of unfamiliar topics.

I would also like to acknowledge that AI was used as an aid in debugging certain issues, such as resolving import errors involving my custom library packages 
(like [commit b167719](https://github.com/ch374nvj/algorithms/commit/b167719e2e8c9b8539bc3affc06efb21cc0b86ed))

## Key Components

### 1. Core Controller Files

- **`pid_control.py`**: Generic PID controller parent class with integrator anti-windup logic
- **`pid_dji_mavic.py`**: Cascaded PID controller specific to DJI Mavic Pro with tuned gains
- **`trim_thrust_calc.py`**: Utility for calculating trim thrust and RPM values

### 2. Interface Layer

- **`vehicle_interface.py`**: Vehicle interface including actuator models and control allocation
- **`frame_transformation.py`**: Coordinate transformation between body frame and NED frame
- **`ctrl_allocator.py`**: Control moment allocation for multi-rotor vehicles
- **`actuator_model.py`**: Detailed actuator model for thrust generation

### 3. Simulation and Testing

- **`main.py`**: Complete nonlinear simulation with disturbance rejection testing
- **`control_linear_model.py`**: Linear model control validation and comparison
- **`tools/plotter.py`**: Plotting utilities for result visualization

### 4. Documentation Structure

```
/Aerospace/control/
├── controller/
│   ├── pid_control.py              # Generic PID controller
│   ├── multirotor/
│   │   └── DJI_Mavic_Pro/
│   │       └── pid_dji_mavic.py   # Mavic Pro cascaded controller
├── interface/
│   ├── multirotor/
│   │   └── DJI_Mavic_Pro/
│   │       ├── vehicle_interface.py  # Vehicle interface
│   │       ├── ctrl_allocator.py    # Control allocation
│   │       └── actuator_model.py   # Actuator model
│   └── frame_transformation.py     # Frame transformation
├── tools/
│   ├── plotter.py                  # Plotting utilities
│   └── __init__.py
├── main.py                         # Full nonlinear simulation
├── control_linear_model.py         # Linear model control
├── trim_thrust_calc.py             # Trim calculation utilities
├── savefig/                       # Generated plots and results
└── README.md                      # This document
```

## Quick Start

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn ussa1976
```

### Running Full Simulation

```bash
cd ./algorithms/Aerospace/control
python main.py
```

This runs a complete nonlinear simulation with:
- Roll, pitch, yaw, and altitude disturbances
- Cascaded PID controllers
- 6-DOF quadcopter dynamics
- Detailed trajectory plots

### Running Linear Model Tests

```bash
cd ./algorithms/Aerospace/control
python control_linear_model.py
```

This runs linear model control validation for:
- Pitch rate control
- Roll rate control  
- Yaw rate control
- Altitude control

on simple rigid model and state-space models derived in the `simulations` submodule

### Trim Thrust Calculation

```bash
cd ./algorithms/Aerospace/control
python trim_thrust_calc.py
```

Calculates hover thrust and corresponding RPM values for:
- Given RPM, compute thrust
- Given thrust, compute RPM

used to compute hover thrust rpm (RPM @ T_hover) for the cascaded  controller.

## Control System Architecture

### Controller Gains

The system uses analytically tuned gains based on first-principles control design:

**Roll, Pitch, Yaw Angle Control (P controllers)**:
- Roll: kp = 2.0
- Pitch: kp = 2.5
- Yaw: kp = 2.0

**Rate Control (PI controllers)**:
- Roll rate (p): kp = 0.00864, ki = 0.3456
- Pitch rate (q): kp = 0.00864, ki = 0.3456
- Yaw rate (r): kp = 0.1008, ki = 0.448

**Altitude Control (P controller)**:
- Altitude: kp = 2.0

**Vertical Velocity Control (PI controller)**:
- Vertical velocity: kp = 10.6992, ki = 47.552

### Hierarchical Control Structure

1. **Outer Loop (Angle Control)**:
   - Input: Desired roll, pitch, yaw angles
   - Output: Rate commands
   - Controllers: P controllers

2. **Inner Loop (Rate Control)**:
   - Input: Rate commands
   - Output: Moment commands
   - Controllers: PI controllers

3. **Altitude Control**:
   - Input: Desired altitude
   - Output: Vertical velocity commands
   - Controllers: P controller for altitude, PI controller for rate

### State-Space Integration

The system supports both nonlinear and linearized plant models:

**Nonlinear Model**: Full 6-DOF dynamics integration


**Linear Model**: Faster simulation for analysis

## Simulation Results

The `savefig/` directory contains generated plots from various test scenarios.

### Linear Model Validation
- `linear/pitch_rate_TF_control_doublet.png`: Pitch rate control
- `linear/pitch_control_SS_step.png`: State-space pitch tracking
- `linear/roll_control_TF.png`: Roll control validation
- `linear/roll_control_step_SS.png`: State-space roll tracking
- `linear/yaw_control_step_SS.png`: Yaw control validation
- `linear/alt_ctrl_rigid_step_with_thrust_constraint.png`: Altitude rigid body
- `linear/alt_ctrl_state_space_with_constraint_2.png`: State-space altitude

## Key Features

### 1. Anti-Windup Protection

```python
PIDController(dt, output_limit=[-10,10], integral_limit=[-10,10])
```

Integral states are bounded to prevent windup during saturation.

### 2. Coordinated Turns

During coordinated turns, the system handles:
- Coupled roll-pitch-yaw dynamics
- Altitude hold maintenance
- Coordinated velocity control


### 3. Atmospheric Modeling

- US Standard Atmosphere 1976 profiles
- Altitude-dependent gravity
- Sea-level and high-altitude operation

### 4. actuator Models

Detailed physics-based modeling, obtained [here](https://github.com/souhaiel1/DJI-Mavic-pro-simplified-modelling-and-simulation), enabling:
- Thrust generation from RPM
- Drag torque coefficients
- Air density compensation


## Testing and Validation

### Disturbance Scenarios
1. **Step Input**: Abrupt changes in attitude commands
2. **Doublet Input**: Roll/pitch rate disturbances
3. **Coordinated Turns**: Combined yaw and roll maneuvers
4. **Altitude Changes**: Vertical trajectory tracking

### Performance Metrics
- **Rise Time**: < 2 seconds for angle responses
- **Overshoot**: < 10% for all channels
- **Settling Time**: < 10 seconds for steady-state accuracy
- **Disturbance Rejection**: > 80% disturbance rejection capability

## Future Enhancements

1. **Adaptive Control**: Gain scheduling for different flight regimes
2. **Robust Control**: H-infinity control for disturbance rejection
3. **Guidance**: GPS waypoint navigation
4. **Navigation**: Integration with IMU and GPS

## License

This project is licensed under the terms of the MIT license.

## Contact

For questions or issues, contact me over [LinkedIn](https://linkedin.com/in/ch374n).