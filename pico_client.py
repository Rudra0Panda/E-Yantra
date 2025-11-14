#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

# Import the action and service definitions
from waypoint_navigation.action import NavToWaypoint
from waypoint_navigation.srv import GetWaypoints


class WayPointClient(Node):

    def __init__(self):
        super().__init__("waypoint_client")

        self.goals = []
        self.goal_index = 0

        # Action Client
        self._action_client = ActionClient(self, NavToWaypoint, "waypoint_navigation")

        # Service Client
        self._service_client = self.create_client(GetWaypoints, "waypoints")
        while not self._service_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for waypoint service...")

        self.req = GetWaypoints.Request()

    # ------------------ SEND GOAL ------------------ #

    def send_goal(self, waypoint):
        goal_msg = NavToWaypoint.Goal()
        goal_msg.waypoint.position.x = waypoint[0]
        goal_msg.waypoint.position.y = waypoint[1]
        goal_msg.waypoint.position.z = waypoint[2]

        self.get_logger().info(f"🚀 Sending waypoint goal: {waypoint}")

        self._action_client.wait_for_server()
        send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    # ------------------ GOAL RESPONSE ------------------ #

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("❌ Goal Rejected!")
            return

        self.get_logger().info("✅ Goal Accepted")

        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    # ------------------ RESULT CALLBACK ------------------ #

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"🏁 Hover Time: {result.hover_time:.2f} seconds")

        self.goal_index += 1
        if self.goal_index < len(self.goals):
            self.send_goal(self.goals[self.goal_index])
        else:
            self.get_logger().info("🎉 All waypoints completed!")

    # ------------------ FEEDBACK ------------------ #

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        pos = feedback.current_position.pose.position
        t = feedback.current_position.header.stamp.sec

        self.get_logger().info(
            f"📡 Feedback → X={pos.x:.2f}, Y={pos.y:.2f}, Z={pos.z:.2f}, Time={t}"
        )

    # ------------------ SERVICE CALL ------------------ #

    def send_request(self):
        self.get_logger().info("📨 Requesting waypoints...")
        return self._service_client.call_async(self.req)

    def receive_goals(self):
        future = self.send_request()
        rclpy.spin_until_future_complete(self, future)
        response = future.result()

        if not response:
            self.get_logger().error("❌ Failed to receive waypoints!")
            return

        self.get_logger().info("📍 Waypoints received successfully.")

        for pose in response.waypoints.poses:
            waypoint = [pose.position.x, pose.position.y, pose.position.z]
            self.goals.append(waypoint)
            self.get_logger().info(f"➡️  Waypoint loaded: {waypoint}")

        if self.goals:
            self.send_goal(self.goals[0])
        else:
            self.get_logger().warn("⚠️ No waypoints in the service response.")


def main(args=None):
    rclpy.init(args=args)
    node = WayPointClient()
    node.receive_goals()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 KeyboardInterrupt! Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
