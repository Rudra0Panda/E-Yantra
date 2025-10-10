#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from swift_msgs.msg import SwiftMsgs
from geometry_msgs.msg import PoseArray
from controller_msg.msg import PIDTune
from error_msg.msg import Error


class Swift_Pico(Node):
    """PID-controlled Swift Pico drone node synchronized with sample_time."""
    def __init__(self):
        super().__init__('pico_controller')

        # Drone state variables [x, y, z]
        self.current_state = [0, 0, 0]
        self.desired_state = [-7, 0, 20]

        # RC commands (all integers)
        self.cmd = SwiftMsgs()
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 2000
        self.cmd.rc_aux1 = 0
        self.cmd.rc_aux2 = 0
        self.cmd.rc_aux3 = 0
        self.cmd.rc_aux4 = 1500
        self.cmd.drone_index = 0

        # PID gains [roll, pitch, throttle]
        self.Kp = [10, 10, 10]
        self.Ki = [10, 10, 10]
        self.Kd = [10, 10, 10]

        # PID auxiliary variables
        self.prev_error = [0, 0, 0]
        self.integral = [0, 0, 0]

        self.min_values = [1000, 1000, 1000]
        self.max_values = [2000, 2000, 2000]

        # PID loop sample time
        self.sample_time = 0.033  # seconds

        # Publishers
        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        self.error_pub = self.create_publisher(Error, '/pos_error', 10)

        # Subscribers
        self.create_subscription(PoseArray, '/whycon/poses', self.whycon_callback, 10)
        self.create_subscription(PIDTune, '/throttle_pid', self.altitude_set_pid, 10)
        self.create_subscription(PIDTune, '/pitch_pid', self.pitch_set_pid, 10)
        self.create_subscription(PIDTune, '/roll_pid', self.roll_set_pid, 10)

        # Arm drone
        self.arm()

        # Timer synchronized with sample_time
        self.create_timer(self.sample_time, self.pid_loop)

    # Arm/disarm
    def disarm(self):
        self.cmd.rc_roll = 1000
        self.cmd.rc_pitch = 1000
        self.cmd.rc_yaw = 1000
        self.cmd.rc_throttle = 1000
        self.cmd.rc_aux4 = 1000
        self.command_pub.publish(self.cmd)

    def arm(self):
        self.disarm()
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1500
        self.cmd.rc_aux4 = 2000
        self.command_pub.publish(self.cmd)

    # Pose callback
    def whycon_callback(self, msg):
        self.current_state[0] = int(msg.poses[0].position.x)
        self.current_state[1] = int(msg.poses[0].position.y)
        self.current_state[2] = int(msg.poses[0].position.z)

    # PID tuning callbacks
    def altitude_set_pid(self, alt):
        self.Kp[2] = int(alt.kp * 0.03)
        self.Ki[2] = int(alt.ki * 0.008)
        self.Kd[2] = int(alt.kd * 0.6)

    def pitch_set_pid(self, pitch):
        self.Kp[1] = int(pitch.kp * 0.03)
        self.Ki[1] = int(pitch.ki * 0.008)
        self.Kd[1] = int(pitch.kd * 0.6)

    def roll_set_pid(self, roll):
        self.Kp[0] = int(roll.kp * 0.03)
        self.Ki[0] = int(roll.ki * 0.008)
        self.Kd[0] = int(roll.kd * 0.6)

    # PID loop called every sample_time
    def pid_loop(self):
        # error = desired - current
        error = [self.desired_state[i] - self.current_state[i] for i in range(3)]

        for i in range(3):
            self.integral[i] += error[i] * self.sample_time
            derivative = (error[i] - self.prev_error[i]) / self.sample_time
            output = int(self.Kp[i] * error[i] + self.Ki[i] * self.integral[i] + self.Kd[i] * derivative)

            # Apply correct signs for axes
            if i == 0:  # roll
                self.cmd.rc_roll = max(min(1500 - output, self.max_values[0]), self.min_values[0])
            elif i == 1:  # pitch
                self.cmd.rc_pitch = max(min(1500 + output, self.max_values[1]), self.min_values[1])
            else:  # throttle
                self.cmd.rc_throttle = max(min(1500 + output, self.max_values[2]), self.min_values[2])

            self.prev_error[i] = error[i]

        # Publish commands
        self.command_pub.publish(self.cmd)

        # Publish error (as integers)
        pos_error = Error()
        pos_error.roll_error = int(error[0])
        pos_error.pitch_error = int(error[1])
        pos_error.throttle_error = int(error[2])
        self.error_pub.publish(pos_error)


def main(args=None):
    rclpy.init(args=args)
    node = Swift_Pico()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
