#!/usr/bin/env python3
"""
ROS2 node 'pico_controller' that holds Swift Pico drone level and position
using PID controllers for throttle (altitude), pitch (X-axis), and roll (Y-axis).

ADDED: Exponential Moving Average (EMA) to smooth the final control commands for
less jerky movement.
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

        # Current position [x, y, z]
        self.current_state = [0.0, 0.0, 0.0]

        # Desired position: control X, Y, and Z via PID
        self.desired_state = [-7.0, 0.0, 20.0]  # Target X, Y, Z

        # Command initialization
        self.cmd = SwiftMsgs()
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1500
        self.cmd.rc_aux4 = 1500

        # PID coefficients [roll(y), pitch(x), throttle(z)]
        # Note: Swapped roll/pitch to match array indices to state indices [x, y, z]
        self.Kp = [3.0, 0.0, 0]  # Kp for [x, y, z]
        self.Ki = [0.0, 0.0, 0]  # Ki for [x, y, z]
        self.Kd = [6.0, 0.0, 0]  # Kd for [x, y, z]

        self.prev_error = [0.0, 0.0, 0.0]
        self.error_sum = [0.0, 0.0, 0.0]

        # RC channel limits
        self.max_values = [2000, 2000, 2000]
        self.min_values = [1000, 1000, 1000]

        # ---------- SMOOTHING PARAMETERS (NEW) ----------
        # Exponential Moving Average (EMA) / Low-pass filter factor (alpha)
        # 0.0 < alpha <= 1.0. Smaller alpha = more smoothing, more latency.
        self.smoothing_factor = 0.5

        # Variables to store the previously smoothed command values
        self.smoothed_roll = 1500.0
        self.smoothed_pitch = 1500.0
        self.smoothed_throttle = 1500.0
        # --------------------------------------------------

        # Control loop timing
        self.sample_time = 0.20
        self.last_time = self.get_clock().now().nanoseconds / 1e9

        # ROS2 publishers
        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        self.pos_error_pub = self.create_publisher(Error, '/pos_error', 10)

        # Subscribers
        self.create_subscription(PoseArray, '/whycon/poses', self.whycon_callback, 1)
        self.create_subscription(PIDTune, '/throttle_pid', self.altitude_set_pid, 1)
        self.create_subscription(PIDTune, '/pitch_pid', self.pitch_set_pid, 1)
        self.create_subscription(PIDTune, '/roll_pid', self.roll_set_pid, 1)  # Added for roll tuning

        # Arm drone and start PID timer
        self.arm()
        self.create_timer(self.sample_time, self.pid)

        self.get_logger().info('Position-Hold Pico Controller Started (PID Active) 🚀')

    # ---------- Arm/Disarm ----------
    def disarm(self):
        self.cmd.rc_roll = 1000
        self.cmd.rc_yaw = 1000
        self.cmd.rc_pitch = 1000
        self.cmd.rc_throttle = 1000
        self.cmd.rc_aux4 = 1000
        self.command_pub.publish(self.cmd)
        self.get_logger().info('Drone disarmed')

    def arm(self):
        self.disarm()
        self.cmd.rc_roll = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_throttle = 1500
        self.cmd.rc_aux4 = 2000
        # Initialize smoothed values on arming (NEW)
        self.smoothed_roll = 1500.0
        self.smoothed_pitch = 1500.0
        self.smoothed_throttle = 1500.0
        self.command_pub.publish(self.cmd)
        self.get_logger().info('Drone armed ✅')

    # ---------- PID tuning callbacks ----------
    def altitude_set_pid(self, msg: PIDTune):
        self.Kp[2] = msg.kp
        self.Ki[2] = msg.ki / 250  # Apply scaling for finer control from GUI
        self.Kd[2] = msg.kd
        self.get_logger().info(f'Altitude PID tuned: Kp={self.Kp[2]}, Ki={self.Ki[2]}, Kd={self.Kd[2]}')

    def pitch_set_pid(self, msg: PIDTune):
        self.Kp[0] = msg.kp
        self.Ki[0] = msg.ki / 250  # Apply scaling for finer control from GUI
        self.Kd[0] = msg.kd
        self.get_logger().info(f'Pitch PID tuned: Kp={self.Kp[0]}, Ki={self.Ki[0]}, Kd={self.Kd[0]}')

    def roll_set_pid(self, msg: PIDTune):
        self.Kp[1] = msg.kp
        self.Ki[1] = msg.ki / 250  # Apply scaling for finer control from GUI
        self.Kd[1] = msg.kd
        self.get_logger().info(f'Roll PID tuned: Kp={self.Kp[1]}, Ki={self.Ki[1]}, Kd={self.Kd[1]}')

    # ---------- WhyCon pose update ----------
    def whycon_callback(self, msg: PoseArray):
        if len(msg.poses) == 0:
            return
        self.current_state[0] = msg.poses[0].position.x
        self.current_state[1] = msg.poses[0].position.y
        self.current_state[2] = msg.poses[0].position.z

    # ---------- PID loop ----------
    def pid(self):
        now = self.get_clock().now().nanoseconds / 1e9
        dt = now - self.last_time
        if dt < self.sample_time:
            return
        if dt <= 0.0:
            dt = self.sample_time

        # --- PID calculations for altitude (Z-axis) ---
        error_z = self.desired_state[2] - self.current_state[2]
        diff_error_z = (error_z - self.prev_error[2]) / dt
        self.error_sum[2] += error_z * dt
        out_z = (
                self.Kp[2] * error_z
                + self.Ki[2] * self.error_sum[2]
                + self.Kd[2] * diff_error_z
        )

        # --- PID calculations for position (X-axis -> Pitch) ---
        error_x = self.desired_state[0] - self.current_state[0]
        diff_error_x = (error_x - self.prev_error[0]) / dt
        self.error_sum[0] += error_x * dt
        out_x = (
                self.Kp[0] * error_x
                + self.Ki[0] * self.error_sum[0]
                + self.Kd[0] * diff_error_x
        )

        # --- PID calculations for position (Y-axis -> Roll) ---
        error_y = self.desired_state[1] - self.current_state[1]
        diff_error_y = (error_y - self.prev_error[1]) / dt
        self.error_sum[1] += error_y * dt
        out_y = (
                self.Kp[1] * error_y
                + self.Ki[1] * self.error_sum[1]
                + self.Kd[1] * diff_error_y
        )

        # Set commands based on PID outputs
        self.cmd.rc_yaw = 1500  # Keep yaw neutral

        # Calculate raw commands from PID output
        raw_throttle = 1500 - out_z
        raw_pitch = 1500 + out_x
        raw_roll = 1500 - out_y

        # ---------- APPLY SMOOTHING (NEW) ----------
        # Apply Exponential Moving Average (EMA) as a low-pass filter
        self.smoothed_throttle = (self.smoothing_factor * raw_throttle) + \
                                 (1 - self.smoothing_factor) * self.smoothed_throttle

        self.smoothed_pitch = (self.smoothing_factor * raw_pitch) + \
                              (1 - self.smoothing_factor) * self.smoothed_pitch

        self.smoothed_roll = (self.smoothing_factor * raw_roll) + \
                             (1 - self.smoothing_factor) * self.smoothed_roll

        # Assign the smoothed values to the final command
        self.cmd.rc_throttle = int(self.smoothed_throttle)
        self.cmd.rc_pitch = int(self.smoothed_pitch)
        self.cmd.rc_roll = int(self.smoothed_roll)
        # ----------------------------------------------

        # Clamp commands to safe RC limits
        self.cmd.rc_throttle = max(self.min_values[2], min(self.max_values[2], self.cmd.rc_throttle))
        self.cmd.rc_pitch = max(self.min_values[0], min(self.max_values[0], self.cmd.rc_pitch))
        self.cmd.rc_roll = max(self.min_values[1], min(self.max_values[1], self.cmd.rc_roll))

        # Publish the final commands
        self.command_pub.publish(self.cmd)

        # Update state for the next loop
        self.prev_error[2] = error_z
        self.prev_error[0] = error_x
        self.prev_error[1] = error_y
        self.last_time = now

        # Updated debug print to include Y-axis and Roll
        self.get_logger().info(
            f"Z: {self.current_state[2]:.2f}, Tgt Z: {self.desired_state[2]:.2f}, Thr: {self.cmd.rc_throttle} | "
            f"X: {self.current_state[0]:.2f}, Tgt X: {self.desired_state[0]:.2f}, Pit: {self.cmd.rc_pitch} | "
            f"Y: {self.current_state[1]:.2f}, Tgt Y: {self.desired_state[1]:.2f}, Rol: {self.cmd.rc_roll}"
        )

    # ---------- Manual control for position ----------
    def increase_altitude(self, delta=1.0):
        self.desired_state[2] += delta
        self.get_logger().info(f"Increased target altitude → {self.desired_state[2]}")

    def decrease_altitude(self, delta=1.0):
        self.desired_state[2] -= delta
        self.get_logger().info(f"Decreased target altitude → {self.desired_state[2]}")

    def move_right(self, delta=1.0):
        self.desired_state[0] += delta
        self.get_logger().info(f"Increased Target X (Right) -> {self.desired_state[0]}")

    def move_left(self, delta=1.0):
        self.desired_state[0] -= delta
        self.get_logger().info(f"Decreased Target X (Left) -> {self.desired_state[0]}")

    def move_forward(self, delta=1.0):
        self.desired_state[1] += delta
        self.get_logger().info(f"Increased Target Y (Forward) -> {self.desired_state[1]}")

    def move_backward(self, delta=1.0):
        self.desired_state[1] -= delta
        self.get_logger().info(f"Decreased Target Y (Backward) -> {self.desired_state[1]}")


def main(args=None):
    rclpy.init(args=args)
    node = Swift_Pico()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt → Shutting down controller.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
