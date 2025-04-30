# Created by Tommaso Tubaldo - 6th March 2025
#
# ROBOT ROUTINES

import math, asyncio
from controller import Robot, Camera, PositionSensor, GPS


class Robot_Routines(Robot):
    def __init__(self):
        super(Robot_Routines, self).__init__()
        self.timestep = int(self.getBasicTimeStep())  # set the control time step
        self.pi = math.pi
        self.wheel_radius = 0.033
        self.wheel_distance = 0.178     # 0.160 from PROTO file, 0.178 through tests

        # Initialization of the Actuators
        #self.max_speed = 6.67
        self.max_speed = 0.4 * 6.67
        self.left_motor = self.getDevice('left wheel motor')
        self.right_motor = self.getDevice('right wheel motor')

        # Initialization of the Position Sensors
        self.left_sensor = PositionSensor('left wheel sensor')
        self.right_sensor = PositionSensor('right wheel sensor')
        self.left_sensor.enable(self.timestep)
        self.right_sensor.enable(self.timestep)

        # Initialization of the GPS
        self.gps = GPS('GPS')
        self.gps.enable(self.timestep)

        # Initialization of the Camera
        self.camera = Camera('CAMERA')
        self.camera.enable(self.timestep)

        self.step(self.timestep)


    def theta2phi(self, theta_r, theta_l):
        """
        Transformation from angular pos/vel of the wheels (theta_r, theta_l) to angular pos/vel of the robot (phi)

        :param theta_r: Angular position/velocity of the right wheel
        :param theta_l: Angular position/velocity of the left wheel
        :return: Angular position/velocity of the robot
        """
        return (self.wheel_radius/self.wheel_distance)*(theta_r-theta_l)


    def phi2theta(self, phi):
        """
        Transformation from angular pos/vel of the robot (phi) to angular pos/vel of the wheels (theta = theta_r = theta_l)

        :param phi: Angular position/velocity of the robot wrt the x-axis
        :return: Angular position/velocity of the wheels
        """
        return (self.wheel_distance/(2*self.wheel_radius))*phi   # (0.08/0.033)*phi


    def lin2ang_wheels(self, linear_pos):
        """
        Transformation from linear to angular position of a wheel

        :param linear_pos: Linear position of the wheel
        :return: Angular position of the wheel
        """
        return linear_pos/self.wheel_radius


    def ang2lin_wheels(self, angular_pos):
        """
        Transformation from angular position to linear position of a wheel

        :param angular_pos: Angular position of the wheel
        :return: Linear position of the wheel
        """
        return angular_pos*self.wheel_radius


    def rot_and_take_images(self):
        """
        Makes the robot do a full rotation while taking several images of the surrounding

        :return: Array containing the images
        """
        print()
        #utils.print_with_ellipsis("Rotating and taking images")
        print("\nRotating and Taking images...")
        # Defining vector of images
        img_vector = []

        # Init Actuators
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.step(self.timestep)
        theta_d = self.phi2theta(self.pi/4)    # Impose ang vel of the robot of pi/4 [rad/s]

        phi = 0
        phi_0 = self.theta2phi(self.right_sensor.getValue(), self.left_sensor.getValue())
        delta_phi = phi - phi_0

        while delta_phi <= 2*self.pi + 1e-1:
            self.left_motor.setVelocity(-theta_d)
            self.right_motor.setVelocity(theta_d)

            theta_r = self.right_sensor.getValue()
            theta_l = self.left_sensor.getValue()
            phi = self.theta2phi(theta_r, theta_l)
            delta_phi = phi - phi_0
            #print(f"Theta R: {theta_r:.2f}","   ",f"Theta L: {theta_l :.2f}","   ",f"Delta Phi: {phi-phi_0:.2f}")

            if (delta_phi  % (self.pi / 4)) < 2e-2:
                self.left_motor.setVelocity(0)
                self.right_motor.setVelocity(0)

                img_vector.append(self.camera.getImage())

                self.left_motor.setVelocity(-theta_d)
                self.right_motor.setVelocity(theta_d)

            self.step(self.timestep)

        return img_vector


    #CAN BE IMPROVED WITH POSITION SENSORS!!
    def get_gps_position(self):
        """Returns the GPS coordinates of the robot (TurtleBot)

        Returns:
            GPS coordinates as a three-dimensional array [x,y,z]
        """
        print("\nGetting GPS position...")
        return self.gps.getValues()


    def get_image(self):
        """Returns an image taken from the robot (TurtleBot) camera

        Returns:
            Camera image in BGRA format
        """
        print("\nGetting image...")
        return self.camera.getImage()


    def set_position(self, right_position, left_position):
        """Sets the absolute linear position of right and left wheel.

        Args:
            right_position: Angular position of the right wheel
            left_position: Angular position of the left wheel
        """
        print("\nSetting wheels position...")
        self.right_motor.setPosition(self.lin2ang_wheels(right_position))
        self.left_motor.setPosition(self.lin2ang_wheels(left_position))
        self.right_motor.setVelocity(0.4 * self.max_speed)
        self.left_motor.setVelocity(0.4 * self.max_speed)


    def set_velocity(self, right_velocity, left_velocity):
        """Sets the linear velocity of right and left wheel.

        Args:
            right_velocity: Angular velocity of the right wheel between [0,1]
            left_velocity: Angular velocity of the left wheel between [0,1]
        """
        print("\nSetting wheels velocity...")
        self.right_motor.setPosition(float('+inf'))
        self.left_motor.setPosition(float('+inf'))
        self.right_motor.setVelocity(right_velocity * self.max_speed)
        self.left_motor.setVelocity(left_velocity * self.max_speed)


    async def sim_loop(self):
        """Continuously step through the simulation."""
        while self.step(self.timestep) != -1:
            await asyncio.sleep(0)