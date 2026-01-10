#!/usr/bin/env python3
"""
Stable ROS2 PID controller for Swift Pico drone.
PID control for Pitch, Roll, Throttle with filtering and smoothing.
"""

import rclpy
from rclpy.node import Node
import time
from rc_msgs.msg import RCMessage
from rc_msgs.srv import CommandBool
from geometry_msgs.msg import PoseArray,Pose
from controller_msg.msg import PIDTune


class Swift_Pico(Node):
    """
    PID controller Node for stabilizing Swift Pico drone using WhyCon.
    """

    def __init__(self):
        super().__init__('pico_controller')

        # -------- States --------
        self.current_state = [0.0, 0.0, 0.0]      # x, y, z
        self.desired_state = [0.0, -5.5, 19.0]    # x, y, z

        # -------- PID Gains [Pitch :x , Roll : y, Throttle : z] --------
        
    
        
        self.Kp = [5.0, 3.0, 8.6]
        self.Ki = [4.0, 7.0, 6.04]
        self.Kd = [5.3, 3.2, 5.0]
        
        #value 1
        
        #self.Kp = [3.0, 2.0, 8.6]
        #self.Ki = [5.0, 5.0, 6.04]
        #self.Kd = [3.3, 1.2, 3.0]

        # -------- PID Internals --------
        self.prev_error = [0.0, 0.0, 0.0]
        self.error_sum = [0.0, 0.0, 0.0]
        self.prev_output = [1500.0, 1500.0, 1500.0]
        self.prev_filtered_derivative = [0.0, 0.0, 0.0]

        # -------- RC Command --------
        self.cmd = RCMessage()
        self.max_values = [2000, 2000, 2000]
        self.min_values = [1000, 1000, 1200]

        # -------- Timing --------
        self.sample_time = 0.022  # ~45 Hz
        self.last_time = self.get_clock().now().nanoseconds / 1e9

        # -------- Filters & Limits --------
        self.output_smoothing_alpha = 0.7
        self.derivative_filter_beta = 0.55
        self.integral_limit = 80.0
        self.deadband = 0.30

        # -------- Publishers --------
        self.command_pub = self.create_publisher(
            RCMessage,
            '/drone/rc_command',
            10
        )

        # -------- Subscribers --------
        self.create_subscription(
            PoseArray,
            '/whycon/poses',
            self.whycon_callback,
            1
        )

        self.create_subscription(
            PIDTune,
            '/pitch_pid',
            self.pitch_set_pid,
            1
        )

        self.create_subscription(
            PIDTune,
            '/roll_pid',
            self.roll_set_pid,
            1
                
        )

        
        self.create_subscription(
            PIDTune,
            '/throttle_pid',
            self.altitude_set_pid,
            1
        )

        
        desiredStatePub = self.create_publisher(
            Pose,
            '/desired_state',
            10
        )
        


       
        # -------- Arming Client --------
        self.cli = self.create_client(CommandBool, '/drone/cmd/arming')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for arming service...')

        self.req = CommandBool.Request()

        self.get_logger().info("🧭 Swift Pico Controller Starting...")
        
        self.arm()

        # -------- Control Loop --------
        time.sleep(0.5)
        self.create_timer(self.sample_time, self.pid_loop)
        self.get_logger().info("✅ Controller Running")

    # ================= ARM / DISARM =================

    
    
    def arm(self):
        self.req.value = True
        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().info("🟢 Drone Armed")

    def disarm(self):
        if not rclpy.ok():
            return

        self.get_logger().info("🔴 Disarming Drone...")
        self.req.value = False
        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().info("🛑 Drone Disarmed")

    # ================= PID TUNING =================

    def altitude_set_pid(self, msg: PIDTune):
        self.Kp[2] = msg.kp / 50.0
        self.Ki[2] = msg.ki / 500.0
        self.Kd[2] = msg.kd / 50.0

        print(
            f'Altitude PID updated: '
            f'Kp={self.Kp[2]}, Ki={self.Ki[2]}, Kd={self.Kd[2]}'
        )

    def pitch_set_pid(self, msg: PIDTune):
        self.Kp[0] = msg.kp / 50.0
        self.Ki[0] = msg.ki / 250.0
        self.Kd[0] = msg.kd / 50.0

        print(
            f'Pitch PID updated: '
            f'Kp={self.Kp[0]}, Ki={self.Ki[0]}, Kd={self.Kd[0]}'
        )

    def roll_set_pid(self, msg: PIDTune):
        self.Kp[1] = msg.kp / 50.0
        self.Ki[1] = msg.ki / 250.0
        self.Kd[1] = msg.kd / 50.0

        print(
            f'Roll PID updated: '
            f'Kp={self.Kp[1]}, Ki={self.Ki[1]}, Kd={self.Kd[1]}'
        )

    def desired_stateCallback(self):
            msg = Pose()
            msg.position.x = self.desired_state[0]
            msg.position.y = self.desired_state[1]
            msg.position.z = self.desired_state[2]
            self.desiredStatePub.publish(msg)
                
    
    # ================= WHYCON =================

    def whycon_callback(self, msg: PoseArray):
        if msg.poses:
            p = msg.poses[0].position
            self.current_state = [p.x, p.y, p.z]

    # ================= PID LOOP =================

   
    
    
    def pid_loop(self):
        now = self.get_clock().now().nanoseconds / 1e9
        dt = now - self.last_time

         
         
            
        if dt <= 0.0:
            return

        self.last_time = now
        outputs = [0.0, 0.0, 0.0]

        for i in range(3):
            error = self.desired_state[i] - self.current_state[i]

            if abs(error) < self.deadband:
                error = 0.0

            # P term
            p_term = self.Kp[i] * error

            # I term
            if abs(error) > self.deadband:
                self.error_sum[i] += error * dt

            self.error_sum[i] = max(
                -self.integral_limit,
                min(self.integral_limit, self.error_sum[i])
            )

            i_term = self.Ki[i] * self.error_sum[i]

            # D term (filtered)
            derivative = (error - self.prev_error[i]) / dt
            filtered = (
                self.derivative_filter_beta * self.prev_filtered_derivative[i]
                + (1.0 - self.derivative_filter_beta) * derivative
            )

            d_term = self.Kd[i] * filtered

            outputs[i] = p_term + i_term + d_term

            self.prev_error[i] = error
            self.prev_filtered_derivative[i] = filtered

        # -------- RC Mixing --------
        rc_pitch = 1420 + outputs[0]
        rc_roll = 1420 + outputs[1]
        rc_throttle = 1400 - outputs[2]

        # -------- Output Smoothing --------
        sp = self.output_smoothing_alpha

        pitch = sp * self.prev_output[0] + (1 - sp) * rc_pitch
        roll = sp * self.prev_output[1] + (1 - sp) * rc_roll
        throttle = sp * self.prev_output[2] + (1 - sp) * rc_throttle

        self.cmd.rc_pitch = int(
            max(self.min_values[0], min(self.max_values[0], pitch))
        )
        self.cmd.rc_roll = int(
            max(self.min_values[1], min(self.max_values[1], roll))
        )
        self.cmd.rc_throttle = int(
            max(self.min_values[2], min(self.max_values[2], throttle))
        )
        self.cmd.rc_yaw = 1500

        self.command_pub.publish(self.cmd)
        self.prev_output = [pitch, roll, throttle]

        # -------- Error Logging --------
        z_error = self.desired_state[2] - self.current_state[2]
        x_error = self.desired_state[0] - self.current_state[0]
        y_error = self.desired_state[1] - self.current_state[1]

        self.get_logger().info(
            f"Z Error: {z_error:.3f} | Throttle: {self.cmd.rc_throttle} | "
            f"X Error: {x_error:.3f} | Y Error: {y_error:.3f} | "
            f"Pitch: {self.cmd.rc_pitch} | Roll: {self.cmd.rc_roll}"
        )

    # ================= CLEANUP =================

    def destroy(self):
        self.disarm()
        super().destroy_node()


# ================= MAIN =================

def main(args=None):
    rclpy.init(args=args)
    node = Swift_Pico()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        if rclpy.ok():
            node.get_logger().info("🛑 User Interrupt → Disarming")
            node.disarm()

    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
