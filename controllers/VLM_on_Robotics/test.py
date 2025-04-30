from Robot_Routines.robot_routines import Robot_Routines

robot = Robot_Routines()

#one_rotation = robot.pi*robot.wheel_distance
one_rotation = robot.pi*0.178

robot.set_position(0,0)
#robot.set_position(one_rotation,-one_rotation)
#robot.set_position(4*one_rotation,-4*one_rotation)

while robot.step(robot.timestep) != -1:
    left_pos = robot.left_sensor.getValue()
    right_pos = robot.right_sensor.getValue()

    left_pos = robot.ang2lin_wheels(left_pos)
    right_pos = robot.ang2lin_wheels(right_pos)

    print(f"left_pos: {left_pos}, right_pos: {right_pos}")