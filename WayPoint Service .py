#!/usr/bin/env python3
"""
Enhanced Waypoint Controller for Swift Pico Drone
-------------------------------------------------
✅ Runs at higher rate (5 Hz) for smoother feedback.
✅ Holds each waypoint within ±0.4 m tolerance.
✅ Includes hysteresis and drift confirmation delay.
✅ Prevents false resets due to single noisy readings.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Vector3
import math
import time


class WayPoints(Node):

    def __init__(self):
        super().__init__('waypoints_service')

        # Publisher for desired state
        self.pub = self.create_publisher(Vector3, '/desired_state', 10)

        # Subscriber for current position
        self.create_subscription(PoseArray, '/whycon/poses', self.pose_callback, 1)

        # Current position
        self.current_pose = [0.0, 0.0, 0.0]

        # Waypoints [x, y, z]
        self.waypoints = [
            [-7.00,  0.00, 29.22],
            [-7.64,  3.06, 29.22],
            [-8.22,  6.02, 29.22],
            [-9.11,  9.27, 29.27],
            [-5.98,  8.81, 29.27],
            [-3.26,  8.41, 29.88],
            [ 0.87,  8.18, 29.05],
            [ 3.93,  7.35, 29.05]
        ]

        # Parameters
        self.error_margin = 0.4      # ±0.4 m stable zone
        self.exit_threshold = 0.55   # leave zone if error > 0.55 m
        self.hold_time = 3.0         # must hold for 3 seconds
        self.rate_hz = 5.0           # run loop at 5 Hz (was 1 Hz)

        # Timer for main loop
        self.timer = self.create_timer(1.0 / self.rate_hz, self.check_and_publish)

        # State variables
        self.current_wp_index = 0
        self.reached_time = None
        self.holding = False
        self.drift_start_time = None

        self.get_logger().info('🚀 Waypoint node started successfully!')

    # ---------------------- CALLBACK ----------------------
    def pose_callback(self, msg):
        if len(msg.poses) > 0:
            pose = msg.poses[0].position
            self.current_pose = [pose.x, pose.y, pose.z]

    # ---------------------- MAIN LOOP ----------------------
    def check_and_publish(self):
        # End condition
        if self.current_wp_index >= len(self.waypoints):
            self.get_logger().info('✅ All waypoints reached!')
            self.destroy_timer(self.timer)
            return

        wp = self.waypoints[self.current_wp_index]
        error = self.distance(self.current_pose, wp)

        # Publish current target to PID controller
        desired_state_msg = Vector3(x=wp[0], y=wp[1], z=wp[2])
        self.pub.publish(desired_state_msg)

        # Print status
        self.get_logger().info(
            f'→ Target WP{self.current_wp_index + 1}: {wp} | '
            f'Error: {error:.2f} | Holding: {self.holding}'
        )

        # ------------------ HOLD / DRIFT LOGIC ------------------
        if error <= self.error_margin:
            # Enter or continue hold
            if not self.holding:
                self.holding = True
                self.reached_time = time.time()
                self.get_logger().info(f'🟢 Holding at waypoint {self.current_wp_index + 1}...')

            # Check hold duration
            elif time.time() - self.reached_time >= self.hold_time:
                self.get_logger().info(f'✅ Waypoint {self.current_wp_index + 1} held stable for {self.hold_time}s.')
                self.current_wp_index += 1
                self.holding = False
                self.reached_time = None
                self.drift_start_time = None

        elif error > self.exit_threshold:
            # Confirm drift for at least 0.5 s before resetting
            if self.holding:
                if self.drift_start_time is None:
                    self.drift_start_time = time.time()
                elif time.time() - self.drift_start_time > 0.5:
                    self.holding = False
                    self.reached_time = None
                    self.drift_start_time = None
                    self.get_logger().warn('⚠️ Drift confirmed — Hold timer reset')
        else:
            # Inside hysteresis band (0.4–0.55 m) → do nothing
            self.drift_start_time = None

    # ---------------------- UTILITIES ----------------------
    @staticmethod
    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 +
                         (p1[1] - p2[1])**2 +
                         (p1[2] - p2[2])**2)


# ---------------------- MAIN ----------------------
def main(args=None):
    rclpy.init(args=args)
    node = WayPoints()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt, shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
