#!/usr/bin/env python3
import time
import math
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

# Action, message, and service types
from waypoint_navigation.action import NavToWaypoint
from swift_msgs.msg import SwiftMsgs
from geometry_msgs.msg import Pose, Vector3
from nav_msgs.msg import Odometry
from controller_msg.msg import PIDTune
from tf_transformations import euler_from_quaternion


class WayPointServer(Node):
    def __init__(self):
        super().__init__("waypoint_server")

        self.action_callback_group = ReentrantCallbackGroup()
        self.timer_callback_group = ReentrantCallbackGroup()
        self.subscriber_callback_group = ReentrantCallbackGroup()

        # Core drone state variables
        # curr_state: [x, y, z, yaw]
        self.curr_state = [0.0, 0.0, 0.0, 0.0]
        # desired_state: [x, y, z, yaw]
        self.desired_state = [0.0, 0.0, 0.0, 0.0]
        
        # Time for derivative calculation
        self.last_time = self.get_clock().now().nanoseconds / 1e9

        # PID variables for [roll, pitch, throttle, yaw]
        self.Kp = [0.0, 0.0, 0.0, 0.0]
        self.Ki = [0.0, 0.0, 0.0, 0.0]
        self.Kd = [0.0, 0.0, 0.0, 0.0]
        self.prev_error = [0.0, 0.0, 0.0, 0.0]
        self.error_sum = [0.0, 0.0, 0.0, 0.0]
        
        # PID output limits (example values, tune as needed)
        self.output_limits = {
            'roll': [-300, 300],
            'pitch': [-300, 300],
            'throttle': [-300, 300],
            'yaw': [-300, 300]
        }
        
        # Base command values (example: 1500 is often neutral)
        self.base_roll = 1500
        self.base_pitch = 1500
        self.base_throttle = 1500
        self.base_yaw = 1500

        # Publishers
        self.command_pub = self.create_publisher(
            SwiftMsgs, 
            "/drone_command", 
            10
        )

        # Subscribers
        # NEW: Subscriber for drone's current state (odometry)
        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom",  # Adjust topic name as needed
            self.odom_callback,
            10,
            callback_group=self.subscriber_callback_group
        )
        
        # NEW: Subscriber for PID gains
        self.pid_sub = self.create_subscription(
            PIDTune,
            "/pid_tuning",  # Adjust topic name as needed
            self.pid_callback,
            10,
            callback_group=self.subscriber_callback_group
        )

        # Action server
        self._action_server = ActionServer(
            self,
            NavToWaypoint,
            "waypoint_navigation",
            execute_callback=self.execute_callback,
            callback_group=self.action_callback_group,
        )
        
        # NEW: Control loop timer (running at 50Hz)
        self.control_timer = self.create_timer(
            0.02,  # 50 Hz
            self.control_loop_callback,
            callback_group=self.timer_callback_group
        )
        
        self.get_logger().info("Waypoint Server has started.")

    # ------------------- SUBSCRIBER CALLBACKS ------------------- #

    def odom_callback(self, msg: Odometry):
        """Update current drone state from odometry."""
        self.curr_state[0] = msg.pose.pose.position.x
        self.curr_state[1] = msg.pose.pose.position.y
        self.curr_state[2] = msg.pose.pose.position.z

        orientation_q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion(
            [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        )
        self.curr_state[3] = yaw

    def pid_callback(self, msg: PIDTune):
        """Update PID gains."""
        self.Kp[0] = msg.kp_roll
        self.Ki[0] = msg.ki_roll
        self.Kd[0] = msg.kd_roll
        
        self.Kp[1] = msg.kp_pitch
        self.Ki[1] = msg.ki_pitch
        self.Kd[1] = msg.kd_pitch

        self.Kp[2] = msg.kp_throttle
        self.Ki[2] = msg.ki_throttle
        self.Kd[2] = msg.kd_throttle

        self.Kp[3] = msg.kp_yaw
        self.Ki[3] = msg.ki_yaw
        self.Kd[3] = msg.kd_yaw

    # ------------------- ACTION SERVER CALLBACK ------------------- #

    def execute_callback(self, goal_handle):
        """Handle received waypoint goal from client."""
        goal_pose = goal_handle.request.waypoint

        # Set desired position
        self.desired_state[0] = goal_pose.position.x
        self.desired_state[1] = goal_pose.position.y
        self.desired_state[2] = goal_pose.position.z
        
        # Set desired yaw (extract from goal's quaternion)
        orientation_q = goal_pose.orientation
        _, _, desired_yaw = euler_from_quaternion(
            [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        )
        self.desired_state[3] = desired_yaw

        self.get_logger().info(f"🚀 New waypoint goal: {self.desired_state}")

        feedback_msg = NavToWaypoint.Feedback()
        result = NavToWaypoint.Result()

        start_time = self.get_clock().now().nanoseconds / 1e9
        self.max_time_inside_sphere = 0.0
        self.time_inside_sphere = 0.0
        
        # Reset PID errors for the new goal
        self.prev_error = [0.0, 0.0, 0.0, 0.0]
        self.error_sum = [0.0, 0.0, 0.0, 0.0]

        while rclpy.ok():
            # Update feedback
            feedback_msg.current_waypoint.pose.position.x = self.curr_state[0]
            feedback_msg.current_waypoint.pose.position.y = self.curr_state[1]
            feedback_msg.current_waypoint.pose.position.z = self.curr_state[2]
            feedback_msg.current_waypoint.header.stamp.sec = int(self.max_time_inside_sphere)
            goal_handle.publish_feedback(feedback_msg)

            # Check proximity
            in_sphere = self.is_drone_in_sphere(
                self.curr_state,
                [self.desired_state[0], self.desired_state[1], self.desired_state[2]],
                0.8,  # tolerance radius in m
            )

            if in_sphere:
                self.time_inside_sphere += 0.1  # Corresponds to time.sleep(0.1)
            else:
                self.time_inside_sphere = 0.0

            if self.time_inside_sphere > self.max_time_inside_sphere:
                self.max_time_inside_sphere = self.time_inside_sphere

            # Goal reached condition
            if self.max_time_inside_sphere >= 3.0:
                self.get_logger().info("✅ Waypoint reached and hovered for 3s.")
                break
                
            # Check for preemption
            if not goal_handle.is_active:
                self.get_logger().info("Goal aborted or preempted.")
                return NavToWaypoint.Result() # Return empty result
                
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info("Goal canceled.")
                return NavToWaypoint.Result()

            time.sleep(0.1) # Loop check at 10Hz

        result.hov_time = self.get_clock().now().nanoseconds / 1e9 - start_time
        goal_handle.succeed()
        return result

    # ------------------- CONTROL LOOP ------------------- #

    def control_loop_callback(self):
        """Main PID control loop."""
        
        # Calculate time delta (dtime)
        current_time = self.get_clock().now().nanoseconds / 1e9
        dtime = current_time - self.last_time
        self.last_time = current_time
        
        if dtime == 0: # Avoid division by zero on first run
            return

        # 1. Calculate Error
        # Note: This is a simple implementation.
        # A real drone controller would convert world-frame error (x, y)
        # into body-frame error (roll, pitch) based on current yaw.
        # For simplicity, we map x-error -> pitch, y-error -> roll.
        
        error = [0.0] * 4
        # Error for Pitch (controls X)
        error[1] = self.desired_state[0] - self.curr_state[0] 
        # Error for Roll (controls Y)
        error[0] = -(self.desired_state[1] - self.curr_state[1]) # Note: negative sign common
        # Error for Throttle (controls Z)
        error[2] = self.desired_state[2] - self.curr_state[2]
        # Error for Yaw
        error[3] = self.desired_state[3] - self.curr_state[3]
        
        # Handle yaw wrapping (-pi to pi)
        if error[3] > math.pi:
            error[3] -= 2 * math.pi
        elif error[3] < -math.pi:
            error[3] += 2 * math.pi

        # 2. Calculate PID terms
        output = [0.0] * 4
        
        for i in range(4):
            # Proportional
            p_term = self.Kp[i] * error[i]
            
            # Integral (with anti-windup)
            self.error_sum[i] += error[i] * dtime
            # Anti-windup: Clamp integral sum if needed (example limits)
            self.error_sum[i] = max(min(self.error_sum[i], 500.0), -500.0) 
            i_term = self.Ki[i] * self.error_sum[i]
            
            # Derivative
            error_diff = (error[i] - self.prev_error[i]) / dtime
            d_term = self.Kd[i] * error_diff
            
            # Update previous error for next loop
            self.prev_error[i] = error[i]
            
            # 3. Sum terms and add base value
            output[i] = p_term + i_term + d_term

        # 4. Create and publish command message
        cmd = SwiftMsgs()
        # Clamp outputs to limits and add base
        cmd.rc_roll = int(self.base_roll + max(min(output[0], self.output_limits['roll'][1]), self.output_limits['roll'][0]))
        cmd.rc_pitch = int(self.base_pitch + max(min(output[1], self.output_limits['pitch'][1]), self.output_limits['pitch'][0]))
        cmd.rc_throttle = int(self.base_throttle + max(min(output[2], self.output_limits['throttle'][1]), self.output_limits['throttle'][0]))
        cmd.rc_yaw = int(self.base_yaw + max(min(output[3], self.output_limits['yaw'][1]), self.output_limits['yaw'][0]))
        
        self.command_pub.publish(cmd)


    # ------------------- HELPERS ------------------- #
    def is_drone_in_sphere(self, drone_pos, goal_pos, radius):
        """Check if drone is inside sphere centered at goal."""
        dx = drone_pos[0] - goal_pos[0]
        dy = drone_pos[1] - goal_pos[1]
        dz = drone_pos[2] - goal_pos[2]
        return (dx * dx + dy * dy + dz * dz) <= (radius * radius)


def main(args=None):
    rclpy.init(args=args)
    node = WayPointServer()
    
    # Use MultiThreadedExecutor to handle all callbacks concurrently
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