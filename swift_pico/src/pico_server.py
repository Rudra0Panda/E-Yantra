#!/usr/bin/env python3
import time
import math
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from waypoint_navigation.action import NavToWaypoint
from swift_msgs.msg import SwiftMsgs
from geometry_msgs.msg import PoseArray, Vector3, Pose
from nav_msgs.msg import Odometry
from error_msg.msg import Error
from tf_transformations import euler_from_quaternion


class WayPointServer(Node):
    def __init__(self):
        super().__init__("waypoint_server")

        # State variables
        self.cmd = SwiftMsgs()
        self.current_state = [0.0, 0.0, 0.0, 0.0]  # x, y, z, yaw
        self.desired_state = [0.0, 0.0, 0.0, 0.0]

        # PID gains
        self.Kp = [10.0, 8.0, 22.0, 0.5]
        self.Ki = [0.30, 0.35, 5.0, 0.05]
        self.Kd = [11.0, 14.0, 40.0, 0.01]

        # PID states
        self.prev_error = [0.0, 0.0, 0.0, 0.0]
        self.error_sum = [0.0, 0.0, 0.0, 0.0]
        self.prev_output = [1500, 1500, 1500, 1500]

        # Other constants
        self.alpha = 0.35
        self.beta = 0.92
        self.integral_limit = 60.0
        self.deadband = 0.20
        self.sample_time = 0.03
        self.last_time = self.get_clock().now().nanoseconds / 1e9

        # Drone state
        self.diff_x = self.diff_y = self.diff_z = self.diff_yaw = 0.0
        self.max_values = [2000, 2000, 2000, 2000]
        self.min_values = [1000, 1000, 1000, 1000]
        self.stable_counter = 0
        self.initial_yaw_set = False
        self.desired_yaw = None

        # Callback groups
        self.action_callback_group = ReentrantCallbackGroup()
        self.timer_callback_group = ReentrantCallbackGroup()
        self.subscriber_callback_group = ReentrantCallbackGroup()

        # Publishers
        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        self.pos_error_pub = self.create_publisher(Error, '/pos_error', 10)

        # Subscriptions
        self.whycon_sub = self.create_subscription(
            PoseArray, "/whycon/poses", self.whycon_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/rotors/odometry", self.odometry_callback, 10
        )

        # Action server
        self._action_server = ActionServer(
            self,
            NavToWaypoint,
            "waypoint_navigation",
            execute_callback=self.execute_callback,
            callback_group=self.action_callback_group
        )

        # Control loop
        self.control_timer = self.create_timer(self.sample_time, self.pid)

        self.get_logger().info("🧭 Stable Swift Pico Controller started (fixed smoothing)")

        # Arm the drone
        self.arm()

    # ------------------- ARM / DISARM -------------------
    def disarm(self):
        self.cmd.rc_roll = self.cmd.rc_pitch = self.cmd.rc_yaw = self.cmd.rc_throttle = 1000
        self.cmd.rc_aux4 = 1000
        self.command_pub.publish(self.cmd)
        self.get_logger().info("Drone disarmed")

    def arm(self):
        self.cmd.rc_roll = self.cmd.rc_pitch = self.cmd.rc_yaw = self.cmd.rc_throttle = 1500
        self.cmd.rc_aux4 = 2000
        self.command_pub.publish(self.cmd)
        self.get_logger().info("Drone armed ✅")

    # ------------------- CALLBACKS -------------------
    def whycon_callback(self, msg: PoseArray):
        if not msg.poses:
            return
        p = msg.poses[0]
        self.current_state[0] = p.position.x
        self.current_state[1] = p.position.y
        self.current_state[2] = p.position.z

        orientation_q = p.orientation
        _, _, yaw = euler_from_quaternion(
            [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        )
        self.current_state[3] = yaw

    def odometry_callback(self, msg: Odometry):
        orientation_q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion(
            [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        )
        self.current_yaw = yaw
        if not self.initial_yaw_set:
            self.desired_yaw = yaw
            self.initial_yaw_set = True
            self.get_logger().info(f"Initial Yaw Baseline set to: {math.degrees(yaw):.2f}°")

    # ------------------- ACTION SERVER -------------------
    def execute_callback(self, goal_handle):
        self.get_logger().info("Received new waypoint goal.")
        goal_pose = goal_handle.request.waypoint

        self.desired_state[0] = goal_pose.position.x
        self.desired_state[1] = goal_pose.position.y
        self.desired_state[2] = goal_pose.position.z

        q = goal_pose.orientation
        _, _, self.desired_state[3] = euler_from_quaternion([q.x, q.y, q.z, q.w])

        feedback_msg = NavToWaypoint.Feedback()
        result = NavToWaypoint.Result()

        start_time = self.get_clock().now().nanoseconds / 1e9
        self.time_inside_sphere = 0.0

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info("Goal canceled by client.")
                return result

            # Feedback (safe access)
            try:
                feedback_msg.current_position.x = self.current_state[0]
                feedback_msg.current_position.y = self.current_state[1]
                feedback_msg.current_position.z = self.current_state[2]
                goal_handle.publish_feedback(feedback_msg)
            except Exception:
                pass

            # Check if drone reached waypoint
            if self.is_drone_in_sphere(self.current_state, self.desired_state, 0.4):
                self.time_inside_sphere += 0.1
            else:
                self.time_inside_sphere = 0.0

            if self.time_inside_sphere >= 3.0:
                self.get_logger().info("✅ Waypoint reached and hovered for 3s.")
                result.hov_time = self.get_clock().now().nanoseconds / 1e9 - start_time
                goal_handle.succeed()
                return result

            time.sleep(0.1)

        return result

    # ------------------- PID LOOP -------------------
    def pid(self):
        now = self.get_clock().now().nanoseconds / 1e9
        dt = max(self.sample_time, now - self.last_time)
        self.last_time = now

        def safe_num(x, default=0.0):
            try:
                return float(x) if math.isfinite(x) else default
            except Exception:
                return default

        def compute_pid(axis, desired, current, prev_err, err_sum, diff_prev):
            error = desired - current
            if axis == 3:
                error = math.atan2(math.sin(error), math.cos(error))
            if abs(error) < self.deadband:
                error = 0.0
            diff_raw = (error - prev_err) / dt if dt > 0 else 0.0
            diff_filtered = self.beta * diff_prev + (1 - self.beta) * diff_raw
            err_sum += error * dt
            err_sum = max(-self.integral_limit, min(self.integral_limit, err_sum))
            output = self.Kp[axis]*error + self.Ki[axis]*err_sum + self.Kd[axis]*diff_filtered
            return output, error, err_sum, diff_filtered

        out_x, ex, self.error_sum[0], self.diff_x = compute_pid(0, self.desired_state[0], self.current_state[0],
                                                                self.prev_error[0], self.error_sum[0], self.diff_x)
        out_y, ey, self.error_sum[1], self.diff_y = compute_pid(1, self.desired_state[1], self.current_state[1],
                                                                self.prev_error[1], self.error_sum[1], self.diff_y)
        out_z, ez, self.error_sum[2], self.diff_z = compute_pid(2, self.desired_state[2], self.current_state[2],
                                                                self.prev_error[2], self.error_sum[2], self.diff_z)
        out_yaw, ez_yaw, self.error_sum[3], self.diff_yaw = compute_pid(3, self.desired_state[3], self.current_state[3],
                                                                        self.prev_error[3], self.error_sum[3], self.diff_yaw)

        rc_thr = int(1500 - out_z)
        rc_pitch = int(1500 + out_x)
        rc_roll = int(1500 - out_y)
        rc_yaw = int(1500 + out_yaw)

        # Clamp
        self.cmd.rc_throttle = max(self.min_values[2], min(self.max_values[2], rc_thr))
        self.cmd.rc_pitch = max(self.min_values[0], min(self.max_values[0], rc_pitch))
        self.cmd.rc_roll = max(self.min_values[1], min(self.max_values[1], rc_roll))
        self.cmd.rc_yaw = max(self.min_values[3], min(self.max_values[3], rc_yaw))

        self.command_pub.publish(self.cmd)

        self.prev_error = [ex, ey, ez, ez_yaw]
        self.prev_output = [rc_pitch, rc_roll, rc_thr, rc_yaw]

    # ------------------- UTILITY -------------------
    def is_drone_in_sphere(self, drone_pos, goal_pos, radius):
        dx = drone_pos[0] - goal_pos[0]
        dy = drone_pos[1] - goal_pos[1]
        dz = drone_pos[2] - goal_pos[2]
        return (dx * dx + dy * dy + dz * dz) <= (radius * radius)


def main(args=None):
    rclpy.init(args=args)
    node = WayPointServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt, shutting down.")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main() 