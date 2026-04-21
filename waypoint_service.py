#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from waypoint_navigation.srv import GetWaypoints


class WayPoints(Node):
    def __init__(self):
        super().__init__('waypoints_service')

        # Create service
        self.srv = self.create_service(GetWaypoints, 'waypoints', self.waypoint_callback)

        # Define static waypoints
        self.waypoints = [
            [-7.00,  0.00, 29.22],
            [-7.64,  3.06, 29.22],
            [-8.22,  6.02, 29.22],
            [-9.11,  9.27, 29.27],
            [-5.98,  8.81, 29.27],
            [-3.26,  8.41, 29.88],
            [0.87,   8.18, 29.05],
            [3.93,   7.35, 29.05]
        ]

    def waypoint_callback(self, request, response):
        if request.get_waypoints:
            # Populate PoseArray
            response.waypoints.poses = [Pose() for _ in range(len(self.waypoints))]

            for i, (x, y, z) in enumerate(self.waypoints):
                response.waypoints.poses[i].position.x = x
                response.waypoints.poses[i].position.y = y
                response.waypoints.poses[i].position.z = z

            self.get_logger().info("Sent waypoints successfully.")
        else:
           response.waypoints.poses = [Pose() for _ in range(len(self.waypoints))]

           for i, (x, y, z) in enumerate(self.waypoints):
                response.waypoints.poses[i].position.x = x
                response.waypoints.poses[i].position.y = y
                response.waypoints.poses[i].position.z = z

        return response


def main():
    rclpy.init()
    node = WayPoints()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt, shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
