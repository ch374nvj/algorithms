# Algorithms Repository

A consolidated repository for implementing and testing control algorithms across multiple engineering domains.

## Domains Currently Implemented

### Aerospace/Control
- **Quadcopter Control Algorithms**: Cascaded PID controllers for attitude and altitude control
- **Nonlinear Simulation**: Full 6-DOF dynamics integration
- **Linear Model Validation**: State-space control analysis
- **Actuator Models**: Detailed physics-based modeling for multi-rotor systems
- **Tools**: Plotting utilities, frame transformations, control allocation

### Simulations
- **Aerospace/Six-DOF**: Rigid body dynamics and linearizations
- **Electrical**: Signal processing algorithms (Phase Locked Loop, etc.)
- **General**: Numerical methods, numerical integration

## Repository Structure

```
/Aerospace/control/
├── controller/          # Control algorithms
│   ├── pid_control.py   # Generic PID controller
│   └── multirotor/      # UAV-specific controllers
│       └── DJI_Mavic_Pro/
├── interface/            # Vehicle interfaces
│   ├── multirotor/      # Multi-rotor specific interfaces
│   │   └── DJI_Mavic_Pro/
├── simulations/         # Dynamics simulations
└── tools/              # Utility libraries

simulations/
├── Aerospace/          # Aerospace dynamics
├── Electrical/         # Signal processing
└── common/            # Shared utilities
```

## Key Features

- **Algorithm Documentation**: Each algorithm includes mathematical foundations, implementation details, and test results
- **Simulation Testing**: Realistic test scenarios with disturbance rejection validation
- **Cross-Domain Reuse**: Common utilities and interfaces for algorithm comparison
- **Performance Metrics**: Quantitative analysis of algorithm performance

## Testing & Validation

Each algorithm domain includes:
- Step response analysis
- Doublet tests
- Performance metrics (rise time, overshoot, steady-state accuracy)
<!-- - Comparative results across different algorithms -->

## Future Development

Expanding into:
- Signal processing algorithms
- Machine learning for control augmentation
- Real-time embedded implementations
- Multi-agent coordination systems

## License

MIT License