#!/usr/bin/env python3

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

        # Subscriber for current position from whycon
        self.create_subscription(PoseArray, '/whycon/poses', self.pose_callback, 1)

        # Initialize current position
        self.current_pose = [0.0, 0.0, 0.0]

        # Define waypoints
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
        self.error_margin = 3      # ±0.4 meters
        self.hold_time = 3.0         # seconds
        self.rate_hz = 1.0           # 1 Hz waypoint check rate

        # Timer to manage the loop
        self.timer = self.create_timer(1.0 / self.rate_hz, self.check_and_publish)

        # Internal state
        self.current_wp_index = 0
        self.reached_time = None
        self.holding = False

        self.get_logger().info('Waypoint node started successfully!')

    # ------------------------- Callbacks -------------------------

    def pose_callback(self, msg):
        if len(msg.poses) > 0:
            self.current_pose[0] = msg.poses[0].position.x
            self.current_pose[1] = msg.poses[0].position.y
            self.current_pose[2] = msg.poses[0].position.z

    # ------------------------- Main Control -------------------------

    def check_and_publish(self):
        if self.current_wp_index >= len(self.waypoints):
            self.get_logger().info('✅ All waypoints reached!')
            self.destroy_timer(self.timer)
            return

        wp = self.waypoints[self.current_wp_index]
        error = self.distance(self.current_pose, wp)

        # Publish current target waypoint
        desired_state_msg = Vector3(x=wp[0], y=wp[1], z=wp[2])
        self.pub.publish(desired_state_msg)

        self.get_logger().info(
            f'→ Target WP{self.current_wp_index+1}: {wp} | '
            f'Error: {error:.2f} | Holding: {self.holding}'
        )

        # Check if within error margin
        if error <= self.error_margin:
            if not self.holding:
                self.holding = True
                self.reached_time = time.time()
                self.get_logger().info(f'🟢 Holding at waypoint {self.current_wp_index+1}...')
            else:
                # Check if held for 3 seconds
                if time.time() - self.reached_time >= self.hold_time:
                    self.get_logger().info(f'✅ Waypoint {self.current_wp_index+1} reached and held for 3s.')
                    self.current_wp_index += 1
                    self.holding = False
        else:
            # If drone drifts away, reset hold timer
            if self.holding:
                self.holding = False
                self.reached_time = None

    @staticmethod
    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 +
                         (p1[1] - p2[1])**2 +
                         (p1[2] - p2[2])**2)


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
