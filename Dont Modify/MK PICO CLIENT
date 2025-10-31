#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

# Import the action and service definitions properly
from waypoint_navigation.action import NavToWaypoint
from waypoint_navigation.srv import GetWaypoints

class WayPointClient(Node):

    def __init__(self):
        super().__init__('waypoint_client')
        self.goals = []
        self.goal_index = 0

        # Create an action client for the 'NavToWaypoint' action
        self._action_client = ActionClient(self, NavToWaypoint, 'waypoint_navigation')
        
        # Create a service client for the 'GetWaypoints' service
        self._service_client = self.create_client(GetWaypoints, 'waypoints')

        while not self._service_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        # Prepare request object
        self.req = GetWaypoints.Request()

    # -------- Action Client Functions -------- #

    def send_goal(self, waypoint):
        # Create a NavToWaypoint goal object
        goal_msg = NavToWaypoint.Goal()
        goal_msg.waypoint.position.x = waypoint[0]
        goal_msg.waypoint.position.y = waypoint[1]
        goal_msg.waypoint.position.z = waypoint[2]

        # Wait for the action server
        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()

        self.get_logger().info(f'Sending waypoint goal: {waypoint}')
        send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        self.get_logger().info('Goal response received from action server.')
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: Hover time = {result.hover_time:.2f}s')

        self.goal_index += 1
        if self.goal_index < len(self.goals):
            self.send_goal(self.goals[self.goal_index])
        else:
            self.get_logger().info('✅ All waypoints have been reached successfully.')

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        x = feedback.current_position.pose.position.x
        y = feedback.current_position.pose.position.y
        z = feedback.current_position.pose.position.z
        t = feedback.current_position.header.stamp.sec
        self.get_logger().info(f'📡 Feedback: Current position → X={x:.2f}, Y={y:.2f}, Z={z:.2f}')
        self.get_logger().info(f'Max time inside sphere: {t}')

    # -------- Service Client Functions -------- #

    def send_request(self):
        # Send the request asynchronously
        self.get_logger().info('Requesting waypoints from service...')
        return self._service_client.call_async(self.req)

    def receive_goals(self):
        future = self.send_request()

        # Spin until the service response arrives
        rclpy.spin_until_future_complete(self, future)
        response = future.result()

        if not response:
            self.get_logger().error('Failed to receive waypoints.')
            return

        self.get_logger().info('Waypoints received successfully.')

        for pose in response.waypoints.poses:
            waypoint = [pose.position.x, pose.position.y, pose.position.z]
            self.goals.append(waypoint)
            self.get_logger().info(f'📍 Waypoint: {waypoint}')

        # Start with the first goal
        if self.goals:
            self.send_goal(self.goals[0])
        else:
            self.get_logger().warn('No waypoints found in service response.')


def main(args=None):
    rclpy.init(args=args)
    waypoint_client = WayPointClient()
    waypoint_client.receive_goals()

    try:
        rclpy.spin(waypoint_client)
    except KeyboardInterrupt:
        waypoint_client.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
        waypoint_client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
