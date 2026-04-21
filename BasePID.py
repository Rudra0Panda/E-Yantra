
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
from collections import deque


class WayPointServer(Node):
    def __init__(self):
        super().__init__("waypoint_server")

        # ------------------- Filters -------------------
        # Butterworth filter coefficients

        # State variables
        self.cmd = SwiftMsgs()
        self.current_state = [0.0, 0.0, 0.0, 0.0]  # x, y, z, yaw
        self.desired_state = [0.0,0.0, 0.0, 0.0]

        # PID gains
        self.Kp = [10, 8, 22.0, 0]
        self.Ki = [0.30, 0.35, 5, 0]
        self.Kd = [10, 14, 35.0, 0]

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
            PoseArray, "/whycon/poses", self.whycon_callback, 10,
            callback_group=self.subscriber_callback_group
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/rotors/odometry", self.odometry_callback, 10,
            callback_group=self.subscriber_callback_group
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
        hover_start_time = None
        max_hover_time = 0.0

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info("Goal canceled by client.")
                return result

            # Feedback
            try:
                feedback_msg.current_position.x = self.current_state[0]
                feedback_msg.current_position.y = self.current_state[1]
                feedback_msg.current_position.z = self.current_state[2]
                goal_handle.publish_feedback(feedback_msg)
                print(
                    f"roll:{self.current_state[0]}, pitch:{self.current_state[1]}, throttle:{self.current_state[2]}"
                )
            except Exception:
                pass

            # Hover detection
            in_sphere = self.is_drone_in_sphere(self.current_state, self.desired_state, 0.4)
            if in_sphere:
                if hover_start_time is None:
                    hover_start_time = self.get_clock().now().nanoseconds / 1e9
                else:
                    hover_duration = (
                            (self.get_clock().now().nanoseconds / 1e9) - hover_start_time
                    )
                    if hover_duration > max_hover_time:
                        max_hover_time = hover_duration
                        self.get_logger().info(f"Hovering... {hover_duration:.2f}s inside sphere.")

                    if hover_duration >= 3.0:
                        total_time = (
                                (self.get_clock().now().nanoseconds / 1e9) - start_time
                        )
                        result.hov_time = hover_duration
                        self.get_logger().info(
                            f"Hovered for {hover_duration:.2f}s (Total time: {total_time:.2f}s)."
                        )
                        goal_handle.succeed()
                        return result
            else:
                if hover_start_time is not None:
                    self.get_logger().info("Drone exited hover sphere.")
                    hover_start_time = None

            time.sleep(0.1)
        return result

    def is_drone_in_sphere(self, drone_pos, sphere_center, radius):
        return (
                (drone_pos[0] - sphere_center[0]) ** 2
                + (drone_pos[1] - sphere_center[1]) ** 2
                + (drone_pos[2] - sphere_center[2]) ** 2
        ) <= radius ** 2

    # ------------------- PID LOOP -------------------
    def pid(self):
        # --- Timestamp & dt ---
        now = self.get_clock().now().nanoseconds / 1e9
        dt = now - self.last_time
        if dt <= 0.0 or dt > 1.0:
            dt = self.sample_time
        self.last_time = now

        # --- Helper ---
        def safe_num(x, default=0.0):
            try:
                if x is None or not math.isfinite(x):
                    return default
                return float(x)
            except Exception:
                return default

        # --- Core PID ---
        def compute_pid(axis, desired, current, prev_err, err_sum, diff_prev):
            error = safe_num(desired - current)
            diff_raw = (error - prev_err) / dt if dt > 0 else 0.0
            diff = 0.7 * diff_prev + 0.3 * diff_raw  # Derivative smoothing

            err_sum += error * dt
            err_sum = max(-self.integral_limit, min(self.integral_limit, err_sum))  # Anti-windup

            output = (
                    self.Kp[axis] * error
                    + self.Ki[axis] * err_sum
                    + self.Kd[axis] * diff
            )

            return output, error, err_sum, diff

        # --- PID for X ---
        out_x, ex, self.error_sum[0], self.diff_x = compute_pid(
            0, self.desired_state[0], self.current_state[0],
            self.prev_error[0], self.error_sum[0], getattr(self, "diff_x", 0.0)
        )

        # --- PID for Y ---
        out_y, ey, self.error_sum[1], self.diff_y = compute_pid(
            1, self.desired_state[1], self.current_state[1],
            self.prev_error[1], self.error_sum[1], getattr(self, "diff_y", 0.0)
        )

        # --- PID for Z (Throttle) ---
        out_z, ez, self.error_sum[2], self.diff_z = compute_pid(
            2, self.desired_state[2], self.current_state[2],
            self.prev_error[2], self.error_sum[2], getattr(self, "diff_z", 0.0)
        )

        # --- PID for Yaw ---
        ez_yaw, out_yaw = 0.0, 0.0
        if self.desired_yaw is not None:
            yaw_error = self.desired_yaw - self.current_yaw
            yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))  # wrap-around
            out_yaw, ez_yaw, self.error_sum[3], self.diff_yaw = compute_pid(
                3, self.desired_yaw, self.current_yaw,
                self.prev_error[3], self.error_sum[3], getattr(self, "diff_yaw", 0.0)
            )
        else:
            self.diff_yaw = 0.0

        # --- RC Mapping ---
        rc_thr = int(1500 - out_z)
        rc_pitch = int(1500 + out_x)
        rc_roll = int(1500 + out_y)
        rc_yaw = int(1500 + out_yaw)

        # --- Clamp RC Outputs ---
        rc_thr = max(self.min_values[2], min(self.max_values[2], rc_thr))
        rc_pitch = max(self.min_values[0], min(self.max_values[0], rc_pitch))
        rc_roll = max(self.min_values[1], min(self.max_values[1], rc_roll))
        rc_yaw = max(self.min_values[3], min(self.max_values[3], rc_yaw))

        # --- Apply RC commands ---
        self.cmd.rc_throttle = rc_thr
        self.cmd.rc_pitch = rc_pitch
        self.cmd.rc_roll = rc_roll
        self.cmd.rc_yaw = rc_yaw

        # --- Publish command ---
        self.command_pub.publish(self.cmd)

        # --- Save state for next iteration ---
        self.prev_error = [ex, ey, ez, ez_yaw]
        self.prev_output = [rc_pitch, rc_roll, rc_thr, rc_yaw]

        # --- Logging every 0.5s ---
        if int(now * 2) % 2 == 0:
            self.get_logger().info(
                f"Z: Target={self.desired_state[2]:.2f}, Current={self.current_state[2]:.2f}, Throttle={rc_thr} | "
                f"Y: Target={self.desired_state[1]:.2f}, Current={self.current_state[1]:.2f}, Roll={rc_roll} | "
                f"X: Target={self.desired_state[0]:.2f}, Current={self.current_state[0]:.2f}, Pitch={rc_pitch}"
            )


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
