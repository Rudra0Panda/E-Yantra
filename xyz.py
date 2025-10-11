#!/usr/bin/env python3
"""
ROS2 node 'pico_controller' that holds Swift Pico drone at a desired position using a PID controller.

Publications:
    /drone_command  (swift_msgs/SwiftMsgs)
    /pos_error      (error_msg/Error)

Subscriptions:
    /whycon/poses   (geometry_msgs/PoseArray)
    /throttle_pid   (controller_msg/PIDTune)
    /pitch_pid      (controller_msg/PIDTune)
    /roll_pid       (controller_msg/PIDTune)

Notes:
 - States and PID parameters are stored in lists: [roll/x, pitch/y, throttle/z]
 - You may need to invert signs on outputs depending on your simulator/vehicle.
"""

import rclpy
from rclpy.node import Node
from swift_msgs.msg import SwiftMsgs
from geometry_msgs.msg import PoseArray
from controller_msg.msg import PIDTune
from error_msg.msg import Error


class Swift_Pico(Node):
    def __init__(self):
        super().__init__('pico_controller')

        # Current position [x, y, z] (updated by whycon callback)
        self.current_state = [0.0, 0.0, 0.0]

        # Desired position / setpoint [x_desired, y_desired, z_desired]
        self.desired_state = [-7.0, 0.0, 20.0]

        # Command message initialization (base neutral values = 1500)
        self.cmd = SwiftMsgs()
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1500
        self.cmd.rc_aux4 = 1500

        # PID coefficients for [roll(x), pitch(y), throttle(z)]
        self.Kp = [0.0, 0.0, 0.0]
        self.Ki = [0.0, 0.0, 0.0]
        self.Kd = [0.0, 0.0, 0.0]

        # PID internals
        self.prev_error = [0.0, 0.0, 0.0]
        self.error_sum = [0.0, 0.0, 0.0]

        # Limits for rc channels [roll, pitch, throttle]
        self.max_values = [2000, 2000, 2000]
        self.min_values = [1000, 1000, 1540]  # throttle default lower limit set to 1540

        # Position error message to publish
        self.pos_error = Error()

        # Sample time for PID loop (seconds)
        self.sample_time = 0.033

        # last_time for timing control
        self.last_time = self.get_clock().now().nanoseconds / 1e9

        # Publishers
        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        self.pos_error_pub = self.create_publisher(Error, '/pos_error', 10)

        # Subscribers
        self.create_subscription(PoseArray, '/whycon/poses', self.whycon_callback, 1)
        self.create_subscription(PIDTune, '/throttle_pid', self.altitude_set_pid, 1)
        self.create_subscription(PIDTune, '/pitch_pid', self.pitch_set_pid, 1)
        self.create_subscription(PIDTune, '/roll_pid', self.roll_set_pid, 1)

        # Arm the drone
        self.arm()

        # Timer to run pid periodically
        self.create_timer(self.sample_time, self.pid)

        self.get_logger().info('pico_controller node started')

    # ------------------ Arming / Disarming ------------------
    def disarm(self):
        self.cmd.rc_roll = 1000
        self.cmd.rc_yaw = 1000
        self.cmd.rc_pitch = 1000
        self.cmd.rc_throttle = 1000
        self.cmd.rc_aux4 = 1000
        self.command_pub.publish(self.cmd)
        self.get_logger().info('Drone disarmed (sent 1000s)')

    def arm(self):
        # simple arm sequence: disarm then set aux4 high (2000)
        self.disarm()
        self.cmd.rc_roll = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_throttle = 1500
        self.cmd.rc_aux4 = 2000
        self.command_pub.publish(self.cmd)
        self.get_logger().info('Drone armed (aux4=2000)')

    # ------------------ PID Tune Callbacks ------------------
    def altitude_set_pid(self, msg: PIDTune):
        self.Kp[2] = msg.kp
        self.Ki[2] = msg.ki
        self.Kd[2] = msg.kd
        self.get_logger().info(f'Altitude PID tuned: Kp={self.Kp[2]}, Ki={self.Ki[2]}, Kd={self.Kd[2]}')

    def pitch_set_pid(self, msg: PIDTune):
        self.Kp[1] = msg.kp
        self.Ki[1] = msg.ki
        self.Kd[1] = msg.kd
        self.get_logger().info(f'Pitch PID tuned: Kp={self.Kp[1]}, Ki={self.Ki[1]}, Kd={self.Kd[1]}')

    def roll_set_pid(self, msg: PIDTune):
        self.Kp[0] = msg.kp
        self.Ki[0] = msg.ki
        self.Kd[0] = msg.kd
        self.get_logger().info(f'Roll PID tuned: Kp={self.Kp[0]}, Ki={self.Ki[0]}, Kd={self.Kd[0]}')

    # ------------------ Whycon callback ------------------
    def whycon_callback(self, msg: PoseArray):
        # Ensure there is at least one pose in the array
        if len(msg.poses) == 0:
            return

        self.current_state[0] = msg.poses[0].position.x
        self.current_state[1] = msg.poses[0].position.y
        self.current_state[2] = msg.poses[0].position.z

        # Optionally log at debug level
        # self.get_logger().debug(f'Whycon: current_state={self.current_state}')
        print(self.current_state[0], self.current_state[1], self.current_state[2])

    # ------------------ PID loop ------------------
    def pid(self):
        now = self.get_clock().now().nanoseconds / 1e9
        dt = now - self.last_time
        if dt < self.sample_time:
            return
        if dt <= 0.0:
            dt = self.sample_time

        error = [
            self.desired_state[0] - self.current_state[0],
            self.desired_state[1] - self.current_state[1],
            self.desired_state[2] - self.current_state[2],
        ]

        # publish position error message
        self.pos_error.pitch_error = error[0]
        self.pos_error.roll_error = error[1]
        self.pos_error.throttle_error = error[2]

        diff_error = [(error[i] - self.prev_error[i]) / dt for i in range(3)]
        self.error_sum = [self.error_sum[i] + error[i] * dt for i in range(3)]

        out = [0.0, 0.0, 0.0]
        for i in range(3):
            out[i] = (
                self.Kp[i] * error[i]
                + self.Ki[i] * self.error_sum[i]
                + self.Kd[i] * diff_error[i]
            )

        self.cmd.rc_roll = int(1500 + out[0])
        self.cmd.rc_pitch = int(1500 + out[1])
        self.cmd.rc_throttle = int(1500 + out[2])
        self.cmd.rc_yaw = 1500  # keep yaw centered

        # Clamp channel values
        self.cmd.rc_roll = max(self.min_values[0], min(self.max_values[0], self.cmd.rc_roll))
        self.cmd.rc_pitch = max(self.min_values[1], min(self.max_values[1], self.cmd.rc_pitch))
        self.cmd.rc_throttle = max(self.min_values[2], min(self.max_values[2], self.cmd.rc_throttle))

        # Publish command and pos_error
        self.command_pub.publish(self.cmd)
        self.pos_error_pub.publish(self.pos_error)

        # update state
        self.prev_error = error.copy()
        self.last_time = now


def main(args=None):
    rclpy.init(args=args)
    node = Swift_Pico()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt: shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
