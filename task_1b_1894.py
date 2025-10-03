#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.srv import SetPen, TeleportAbsolute
import time


class myNode(Node):
    def __init__(self):
        super().__init__('circle_drawing_node')
        self.get_logger()
        self.cmd_vel_publisher = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.drawCircle((2, 1))
        self.drawCircle((2, 7))
        self.drawCircle((8, 1))   
        self.drawCircle((8, 7))
        self.drawLine((2, 2), (4, 4))
        self.drawLine((8, 2), (6, 4))
        self.drawLine((8, 8), (6, 6))
        self.drawLine((2, 8), (4, 6))
        self.drawLine((5, 7), (7, 5))
        self.drawLine((7, 5), (5, 3))
        self.drawLine((5, 3), (3, 5))
        self.drawLine((3, 5), (5, 7))
        self.setPenState('off')
        self.teleport((5,5))

    def setPenState(self, state):
        client = self.create_client(SetPen, '/turtle1/set_pen')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger()

        req = SetPen.Request()
        req.off = 1 if state == 'off' else 0  

        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        self.get_logger()

    def teleport(self, pos):
        client = self.create_client(TeleportAbsolute, '/turtle1/teleport_absolute')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger()

        x, y = pos
        req = TeleportAbsolute.Request()
        req.x = float(x)
        req.y = float(y)
        req.theta = 0.0

        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        self.get_logger()

    def drawCircle(self, pos):
        self.setPenState('off')
        time.sleep(1)
        self.teleport(pos)
        time.sleep(1)
        self.setPenState('on')
        vel = Twist()
        vel.linear.x = 2.0
        vel.angular.z = 2.0

        self.get_logger()
        start_time = time.time()
        while time.time() - start_time < 3.5: 
            self.cmd_vel_publisher.publish(vel)
            time.sleep(0.1)

        self.cmd_vel_publisher.publish(Twist())

    def drawLine(self, pos1, pos2):
        self.setPenState('off')
        time.sleep(1)
        self.teleport(pos1)
        time.sleep(1)
        self.setPenState('on')
        time.sleep(1)
        self.teleport(pos2)
        self.get_logger()


def main(args=None):
    rclpy.init(args=args)
    node = myNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
