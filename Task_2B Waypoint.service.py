#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from waypoint_navigation.srv import GetWaypoints


class FullMissionWaypointService(Node):
    def __init__(self):
        super().__init__("waypoints_service_task2b_final")
        self.srv = self.create_service(GetWaypoints, "waypoints", self.waypoint_callback)
        self.get_logger().info("🧭 Full Mission Waypoint Service (with Hover + Spray) Ready")

    def waypoint_callback(self, request, response):
        # --- ensure the client explicitly requested waypoints ---
        if not request.get_waypoints:
            self.get_logger().warn("⚠️ Request received but get_waypoints=False — returning empty list.")
            return response

        wp = []

        # --- 1. Ground to Pesticide Station 2 ---
        wp += [[-6.9, 0.09, 32.16], [-7.0, 0.0, 30.22], [-7.49, -2.83, 30.22],
               [-7.82, -5.55, 30.22], [-8.62, -8.6, 29.6], [-8.53, -8.52, 30.64]]

        # --- 2. Ground to Pesticide Station 1 ---
        wp += [[-7.0, 0.0, 30.22], [-7.64, 3.06, 30.22],
               [-8.22, 6.02, 30.22], [-9.11, 9.27, 31.27], [-9.07, 9.23, 32.57]]

        # --- 3. Pesticide Station 1 to Block 1 ---
        wp += [[-5.976, 8.810, 31.27], [-3.26, 8.41, 29.88], [0.87, 8.18, 29.05],
               [3.929, 7.347, 29.05], [6.60, 6.62, 30.20]]

        # --- 4. Block 1 to Block 2 ---
        wp += [[6.7375, 3.3200, 30.6450], [6.8750, -0.1900, 30.0900],
               [7.0125, -3.5950, 31.5350], [7.15, -7.00, 31.98]]

        # --- 5. Block 2 to Pesticide Station 2 ---
        wp += [[4.0, -7.62, 30.59], [0.8, -8.04, 29.20], [-3.26, -8.12, 29.48],
               [-5.94, -8.36, 29.54], [-8.62, -8.6, 29.6]]

        # --- 6. Block 1 Plant Hover + Spray ---
        wp += [
            [4.39, 8.98, 29.29], [4.5, 8.88, 30.59],      # P1A hover + spray
            [6.45, 8.79, 28.67], [6.85, 8.89, 30.62],     # P1B
            [8.62, 8.80, 28.69], [9.13, 8.80, 30.40],     # P1C
            [4.37, 4.57, 29.19], [4.53, 4.57, 30.79],     # P1D
            [6.76, 4.49, 28.77], [6.45, 4.57, 30.79],     # P1E
            [8.7, 4.53, 28.97],  [9.1, 4.53, 28.97],      # P1F
        ]

        # --- 7. Block 2 Plant Hover + Spray ---
        wp += [
            [4.43, -4.30, 29.74], [4.43, -4.30, 31.1],    # P2A
            [6.68, -4.30, 29.74], [7.05, -4.30, 31.52],   # P2B
            [8.93, -4.30, 29.74], [9.32, -4.30, 31.06],   # P2C
            [4.58, -8.78, 30.62], [4.53, -8.59, 30.76],   # P2D
            [6.76, -8.61, 30.03], [7.05, -8.81, 31.52],   # P2E
            [9.17, -8.61, 30.03], [9.08, -8.92, 31.93],   # P2F
        ]

        # --- 8. Return to Ground Station ---
        wp += [[-5.94, -8.36, 29.54], [-3.26, -8.12, 29.48],
               [0.8, -8.04, 29.20], [-7.82, -5.55, 30.22],
               [-7.49, -2.83, 30.22], [-6.9, 0.09, 32.16]]

        # Populate response
        response.waypoints.poses = [Pose() for _ in range(len(wp))]
        for i, (x, y, z) in enumerate(wp):
            response.waypoints.poses[i].position.x = x
            response.waypoints.poses[i].position.y = y
            response.waypoints.poses[i].position.z = z

        self.get_logger().info(f"✅ Sent {len(wp)} waypoints including hover & spray positions.")
        return response


def main():
    rclpy.init()
    node = FullMissionWaypointService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Service stopped.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
