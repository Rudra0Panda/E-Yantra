#!/usr/bin/env python3
"""
Stable ROS2 PID controller for Swift Pico drone.
Fixed smoothing/filter values for consistent tuning and minimal oscillation.
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

        # --- Desired vs Current ---
        self.current_state = [0.0, 0.0, 0.0]
        self.desired_state = [-7.0, 0.0, 19.0]

        # --- PID constants (unchanged) ---
        self.Kp = [0.0, 0.0, 32]
        self.Ki = [0.0, 0.0, 0]
        self.Kd = [0.0, 0.0, 58]

        # --- Internal PID state ---
        self.prev_error = [0.0, 0.0, 0.0]
        self.error_sum = [0.0, 0.0, 0.0]
        self.prev_output = [1500, 1500, 1500]

        # --- RC Command ---
        self.cmd = SwiftMsgs()
        self.reset_rc()

        # --- RC limits ---
        self.max_values = [2000, 2000, 2000]
        self.min_values = [1000, 1000, 1000]

        # --- Control Timing ---
        self.sample_time = 0.03  # 25 
        self.last_time = self.get_clock().now().nanoseconds / 1e9

        # --- Fixed filter/smoothing constants ---
        self.alpha = 0.20 # output smoothing (higher = smoother, slower)
        self.beta = 0.80   # derivative low-pass filter
        self.integral_limit = 80.0
        self.deadband = 0.10

        # --- Publishers ---
        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        self.pos_error_pub = self.create_publisher(Error, '/pos_error', 10)

        # --- Subscribers ---
        self.create_subscription(PoseArray, '/whycon/poses', self.whycon_callback, 1)
        self.create_subscription(PIDTune, '/throttle_pid', self.altitude_set_pid, 1)
        self.create_subscription(PIDTune, '/pitch_pid', self.pitch_set_pid, 1)
        self.create_subscription(PIDTune, '/roll_pid', self.roll_set_pid, 1)

        # --- Arm and start control ---
        self.arm()
        self.create_timer(self.sample_time, self.pid)
        self.get_logger().info("🧭 Stable Swift Pico Controller started (fixed smoothing)")

    # ---------------- Arm/Disarm ----------------
    def disarm(self):
        self.cmd.rc_roll = self.cmd.rc_pitch = self.cmd.rc_yaw = self.cmd.rc_throttle = 1000
        self.cmd.rc_aux4 = 1000
        self.command_pub.publish(self.cmd)
        self.get_logger().info("Drone disarmed")

    def arm(self):
        self.disarm()
        self.cmd.rc_roll = self.cmd.rc_pitch = self.cmd.rc_yaw = self.cmd.rc_throttle = 1500
        self.cmd.rc_aux4 = 2000
        self.command_pub.publish(self.cmd)
        self.get_logger().info("Drone armed ✅")

    def reset_rc(self):
        self.cmd.rc_roll = self.cmd.rc_pitch = self.cmd.rc_yaw = self.cmd.rc_throttle = 1500
        self.cmd.rc_aux4 = 1500

    # ---------------- PID tuning updates ----------------
    def altitude_set_pid(self, msg):
        self.Kp[2] = msg.kp
        self.Ki[2] = msg.ki / 250
        self.Kd[2] = msg.kd

    def pitch_set_pid(self, msg):
        self.Kp[0] = msg.kp
        self.Ki[0] = msg.ki / 250
        self.Kd[0] = msg.kd

    def roll_set_pid(self, msg):
        self.Kp[1] = msg.kp
        self.Ki[1] = msg.ki / 250
        self.Kd[1] = msg.kd

    # ---------------- Pose callback ----------------
    def whycon_callback(self, msg: PoseArray):
        if msg.poses:
            pose = msg.poses[0].position
            self.current_state = [pose.x, pose.y, pose.z]

    # ---------------- PID loop ----------------
    def pid(self):
        now = self.get_clock().now().nanoseconds / 1e9
        dt = max(self.sample_time, now - self.last_time)
        self.last_time = now

        # PID compute function
        def compute_pid(axis, desired, current, prev_err, err_sum, diff_prev):
            error = desired - current

            # Apply deadband (ignore tiny noise near setpoint)
            if abs(error) < self.deadband:
                error = 0.0

            # Derivative term with low-pass filter
            diff_raw = (error - prev_err) / dt
            diff_filtered = self.beta * diff_prev + (1 - self.beta) * diff_raw

            # Integral with anti-windup
            if abs(error) > self.deadband:
                err_sum += error * dt
            err_sum = max(-self.integral_limit, min(self.integral_limit, err_sum))

            # PID output
            output = (self.Kp[axis] * error +
                      self.Ki[axis] * err_sum +
                      self.Kd[axis] * diff_filtered)
            return output, error, err_sum, diff_filtered

        # Compute for each axis
        out_x, ex, self.error_sum[0], self.diff_x = compute_pid(0, self.desired_state[0], self.current_state[0],
                                                                self.prev_error[0], self.error_sum[0], getattr(self, "diff_x", 0.0))
        out_y, ey, self.error_sum[1], self.diff_y = compute_pid(1, self.desired_state[1], self.current_state[1],
                                                                self.prev_error[1], self.error_sum[1], getattr(self, "diff_y", 0.0))
        out_z, ez, self.error_sum[2], self.diff_z = compute_pid(2, self.desired_state[2], self.current_state[2],
                                                                self.prev_error[2], self.error_sum[2], getattr(self, "diff_z", 0.0))

        # RC mapping
        rc_thr = 1500 - out_z
        rc_pitch = 1500 + out_x
        rc_roll = 1500 - out_y

        # Fixed smoothing (no adaptive alpha)
        rc_thr = int(self.alpha * self.prev_output[2] + (1 - self.alpha) * rc_thr)
        rc_pitch = int(self.alpha * self.prev_output[0] + (1 - self.alpha) * rc_pitch)
        rc_roll = int(self.alpha * self.prev_output[1] + (1 - self.alpha) * rc_roll)

        # Clamp RC values
        self.cmd.rc_throttle = max(self.min_values[2], min(self.max_values[2], rc_thr))
        self.cmd.rc_pitch = max(self.min_values[0], min(self.max_values[0], rc_pitch))
        self.cmd.rc_roll = max(self.min_values[1], min(self.max_values[1], rc_roll))
        self.cmd.rc_yaw = 1500

        self.command_pub.publish(self.cmd)

        # Save for next loop
        self.prev_error = [ex, ey, ez]
        self.prev_output = [rc_pitch, rc_roll, rc_thr]

        # Log periodically
        if int(now * 2) % 2 == 0:
            self.get_logger().info(
                f"Z:{self.current_state[2]:.2f}/{self.desired_state[2]:.2f} Thr:{self.cmd.rc_throttle} | "
                f"X:{self.current_state[0]:.2f}/{self.desired_state[0]:.2f} Pit:{self.cmd.rc_pitch} | "
                f"Y:{self.current_state[1]:.2f}/{self.desired_state[1]:.2f} Rol:{self.cmd.rc_roll}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = Swift_Pico()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Controller stopped.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
