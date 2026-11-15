from interface.multirotor.DJI_Mavic_Pro.actuator_model import ActuatorModel

act_mod = ActuatorModel()

print(act_mod.thrust_from_rpm(11197, 1.2))
print(act_mod.rpm_from_thrust(7.3, 1.2))
print(act_mod.rpm_from_thrust(1.884, 1.2))