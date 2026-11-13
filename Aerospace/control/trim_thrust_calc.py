from interface.multirotor.DJI_Mavic_Pro.actuator_model import ActuatorModel

act_mod = ActuatorModel()

print(act_mod.thrust_from_rpm(5700, 1.2))