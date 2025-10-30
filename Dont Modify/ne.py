#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Vector3

class WayPoints(Node):

    def __init__(self):
        super().__init__('waypoints_service')
        self.pub = self.create_publisher(Vector3, '/desired_state', 10)
        self.waypoints = [
            [-7.00,  0.00, 29.22],  # wp1
            [-7.64,  3.06, 29.22],  # wp2
            [-8.22,  6.02, 29.22],  # wp3
            [-9.11,  9.27, 29.27],  # wp4
            [-5.98,  8.81, 29.27],  # wp5
            [-3.26,  8.41, 29.88],  # wp6
            [ 0.87,  8.18, 29.05],  # wp7
            [ 3.93,  7.35, 29.05]   # wp8
            ]
        self.create_subscription(Pose,'/whycon/poses', self.pose_callback,1)
        self.current_pose = [0.0, 0.0, 0.0]
        
    def pose_callback(self, msg):
        self.current_pose[0] = msg.position.x
        self.current_pose[1] = msg.position.y
        self.current_pose[2] = msg.position.z

    def publish_waypoints(self):
        rate = self.create_rate(0.05)  # 0.5 Hz
        for waypoint in self.waypoints:
            desired_state_msg = Vector3()
            desired_state_msg.x = waypoint[0]
            desired_state_msg.y = waypoint[1]
            desired_state_msg.z = waypoint[2]
            self.pub.publish(desired_state_msg)
            self.get_logger().info(f'Published waypoint: {waypoint}')
            rate.sleep()    

def main():

if __name__ == '__main__':
    main()
        

        