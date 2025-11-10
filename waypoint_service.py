#!/usr/bin/env python3
"""
Enhanced Waypoint Controller for Swift Pico Drone (Updated)
----------------------------------------------------------
✅ Fixed hysteresis (exit > enter) to avoid premature hold resets
✅ Pose timeout to prevent thrust sag when tracking drops
✅ EMA smoothing of WhyCon pose to reduce jitter
✅ Altitude guardian: no descent until close in XY
✅ Descent rate limiting to avoid sudden sinks
✅ Faster publish (10 Hz) for fresher setpoints
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Vector3
import math
import time


class WayPoints(Node):
    # ---------------------- CALLBACK ----------------------
    def __init__(self):
        super().__init__('waypoints_service')

        # Publisher for desired state
        self.pub = self.create_publisher(Vector3, '/desired_state', 10)

        # Subscriber for current position
        self.create_subscription(PoseArray, '/whycon/poses', self.pose_callback, 1)

        # --- Pose handling ---
        self.alpha = 0.25                # EMA smoothing factor
        self.current_pose = [0.0, 0.0, 0.0]
        self.filtered_pose = [0.0, 0.0, 0.0]
        self.last_pose_time = None
        self.pose_timeout_s = 0.5        # stale if no pose within 0.5 s

        # Waypoints [x, y, z]  (UNCHANGED)
        self.waypoints = [
            # 1. Ground Station to Pesticide Station 2
            [-6.9, 0.09, 32.16],             # ground_station:
            [-7.0, 0.0, 30.22],              # hover
            [-7.49, -2.83, 30.22],           # wp1
            [-7.82, -5.55, 30.22],           # wp2
            [-8.62, -8.6, 29.6],             # destination
            [-8.53, -8.52, 30.64],           # pickup

            # 2. Ground Station to Pesticide Station 1
            [-6.9, 0.09, 32.16],             # ground_station:
            [-7.0, 0.0, 30.22],              # hover:
            [-7.64, 3.06, 30.22],            # wp1:
            [-8.22, 6.02, 30.22],            # wp2:
            [-9.11, 9.27, 31.27],            # destination:
            [-9.07, 9.23, 32.57],            # pickup:

            # 3. Pesticide Station 1 to Block 1
            [-9.11, 9.27, 31.27],            # start:
            [-5.976, 8.810, 31.27],          # wp1:
            [-3.26, 8.41, 29.88],            # hula_hoop_2_entry:
            [0.87, 8.18, 29.05],             # hula_hoop_2_exit:
            [3.929, 7.347, 29.05],           # wp2:
            [6.60, 6.62, 30.20],             # block_1:

            # 4. Plant Hover Sequence - Block 1
            [4.39, 8.98, 29.29],             # P1A
            [6.45, 8.79, 28.67],             # P1B
            [8.62, 8.80, 28.69],             # P1C
            [4.37, 4.57, 29.19],             # P1D
            [6.76, 4.49, 28.77],             # P1E
            [8.7, 4.53, 28.97],              # P1F

            # 4. Block 1 to Block 2
            [6.60, 6.62, 30.20],             # block_1:
            [6.7375, 3.3200, 30.6450],       # wp1:
            [6.8750, -0.1900, 30.0900],      # wp2:
            [7.0125, -3.5950, 31.5350],      # block_2:

            # Hover(Plants) Positions:
            [4.43, -4.30, 29.74],            # P2A:
            [6.68, -4.30, 29.74],            # P2B:
            [8.93, -4.30, 29.74],            # P2C:
            [4.58, -8.78, 30.62],            # P2D:
            [6.76, -8.61, 30.03],            # P2E:
            [9.17, -8.61, 30.03],            # P2F:

            # 5. Block 2 to Pesticide Station 2
            [7.15, -7.00, 31.98],            # block_2:
            [4.0, -7.62, 30.59],             # wp1:
            [0.8, -8.04, 29.20],             # hula_hoop_1_exit:
            [-3.26, -8.12, 29.48],           # hula_hoop_1_entry:
            [-5.94, -8.36, 29.54],           # wp2:
            [-8.62, -8.6, 29.6],             # destination:

            # 8. Return Path to Ground Station
            [7.15, -7.00, 31.98],            # block_2:
            [-5.94, -8.36, 29.54],           # wp1:
            [-3.26, -8.12, 29.48],           # hula_hoop_1_entry:
            [0.8, -8.04, 29.20],             # hula_hoop_1_exit:
            [-7.82, -5.55, 30.22],           # wp2:
            [-7.49, -2.83, 30.22],           # hover
            [-6.9, 0.09, 32.16]              # destination:
        ]

        # --- Hold/hysteresis ---
        self.error_margin = 0.40          # enter/keep-hold threshold
        self.exit_threshold = 0.60         # must be > error_margin
        self.hold_time = 2.0               # must hold for 2 seconds

        # --- Loop rate ---
        self.rate_hz = 10.0                # run loop at 10 Hz
        self.timer = self.create_timer(1.0 / self.rate_hz, self.check_and_publish)

        # --- State variables ---
        self.current_wp_index = 0
        self.reached_time = None
        self.holding = False
        self.drift_start_time = None

        # --- Altitude guardian ---
        self.xy_gate_m = 1.2               # only allow descent when XY error <= this
        self.max_descent_rate = 0.6        # m/s downward limit
        self.last_cmd_z = None
        self.last_cmd_time = time.time()

        self.get_logger().info('🚀 Waypoint node started successfully with altitude guardian.')

    def pose_callback(self, msg):
        if msg.poses:
            p = msg.poses[0].position
            self.current_pose = [p.x, p.y, p.z]
            # EMA smoothing
            if self.last_pose_time is None:
                self.filtered_pose = self.current_pose[:]  # init
            else:
                for i in range(3):
                    self.filtered_pose[i] = (
                        self.alpha * self.current_pose[i] +
                        (1.0 - self.alpha) * self.filtered_pose[i]
                    )
            self.last_pose_time = time.time()

    # ---------------------- MAIN LOOP ----------------------
    def check_and_publish(self):
        now = time.time()

        # End condition
        if self.current_wp_index >= len(self.waypoints):
            self.get_logger().info('✅ All waypoints reached!')
            self.destroy_timer(self.timer)
            return

        # Pose freshness
        pose_fresh = (self.last_pose_time is not None) and ((now - self.last_pose_time) <= self.pose_timeout_s)

        wp = self.waypoints[self.current_wp_index]
        wx, wy, wz = wp

        # Decide setpoint with altitude guardian
        if pose_fresh:
            fx, fy, fz = self.filtered_pose
            xy_err = math.hypot(fx - wx, fy - wy)

            # Gate descent: if far in XY, don't command a lower Z than current filtered Z
            if xy_err > self.xy_gate_m:
                z_target = max(fz, wz)   # can climb, but won't descend yet
            else:
                z_target = wz

            # Descent-rate limiter
            if self.last_cmd_z is None:
                self.last_cmd_z = z_target
                self.last_cmd_time = now
            dt = max(1e-3, now - self.last_cmd_time)
            max_down_step = self.max_descent_rate * dt
            if z_target < self.last_cmd_z:
                z_target = max(z_target, self.last_cmd_z - max_down_step)

            desired_state_msg = Vector3(x=wx, y=wy, z=z_target)
            self.last_cmd_z = z_target
            self.last_cmd_time = now

            # For hold logic, compute error to the true waypoint (not the z_target)
            error = self.distance([fx, fy, fz], wp)

        else:
            # Stale pose — keep last commanded Z (don’t advance, don’t evaluate)
            if self.last_cmd_z is None:
                self.last_cmd_z = wz
                self.last_cmd_time = now
            desired_state_msg = Vector3(x=wx, y=wy, z=self.last_cmd_z)
            self.holding = False
            self.drift_start_time = None
            self.reached_time = None
            self.pub.publish(desired_state_msg)
            self.get_logger().warn('⏸️ Pose stale — holding setpoint, not evaluating progress.')
            return

        # Publish current target to PID controller
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
                self.reached_time = now
                self.get_logger().info(f'🟢 Holding at waypoint {self.current_wp_index + 1}...')
            # Check hold duration
            elif now - self.reached_time >= self.hold_time:
                self.get_logger().info(f'✅ Waypoint {self.current_wp_index + 1} held stable for {self.hold_time}s.')
                self.current_wp_index += 1
                self.holding = False
                self.reached_time = None
                self.drift_start_time = None

        elif error > self.exit_threshold:
            # Confirm drift for at least 0.5 s before resetting
            if self.holding:
                if self.drift_start_time is None:
                    self.drift_start_time = now
                elif now - self.drift_start_time > 0.5:
                    self.holding = False
                    self.reached_time = None
                    self.drift_start_time = None
                    self.get_logger().warn('⚠️ Drift confirmed — Hold timer reset')
        else:
            # Inside hysteresis band (0.40–0.60 m) → do nothing
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
