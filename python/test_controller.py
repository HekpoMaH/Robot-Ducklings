#!/usr/bin/env python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import rospy
import sys

# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist
# Occupancy grid.
from nav_msgs.msg import OccupancyGrid
# Position.
from tf import TransformListener
# Goal.
from geometry_msgs.msg import PoseStamped
# Path.
from nav_msgs.msg import Path
# For pose information.
from tf.transformations import euler_from_quaternion

FREE = 0
UNKNOWN = 1
OCCUPIED = 2

ROBOT_RADIUS = 0.105 / 2.

X = 0
Y = 1
YAW = 2

LEADER = 'tb3_0'
FOLLOWER_1 = 'tb3_1'
FOLLOWER_2 = 'tb3_2'

def feedback_linearized(pose, velocity, epsilon):
  u = 0.  # [m/s]
  w = 0.  # [rad/s] going counter-clockwise.

  # MISSING: Implement feedback-linearization to follow the velocity
  # vector given as argument. Epsilon corresponds to the distance of
  # linearized point in front of the robot.

  u = velocity[X] * np.cos(pose[YAW]) + velocity[Y] * np.sin(pose[YAW])
  w = (1 / epsilon) * (-velocity[X] * np.sin(pose[YAW]) + velocity[Y] * np.cos(pose[YAW]))

  return u, w
        
class SLAM(object):
  def __init__(self):
    rospy.Subscriber('/map', OccupancyGrid, self.callback)
    self._tf = TransformListener()
    self._occupancy_grid = None
    self._pose = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    
  def callback(self, msg):
    pass

  def update(self):
    # Get pose w.r.t. map.
    r0_pose = self.get_pose(LEADER)
    r1_pose = self.get_pose(FOLLOWER_1)
    r2_pose = self.get_pose(FOLLOWER_2)

    print('leader')
    print('\t X: ', r0_pose[X])
    print('\t Y: ', r0_pose[Y])
    print('\t YAW: ', r0_pose[YAW])

    print()

    print('follower 1')
    print('\t X: ', r1_pose[X])
    print('\t Y: ', r1_pose[Y])
    print('\t YAW: ', r1_pose[YAW])

    print()

    print('follower 2')
    print('\t X: ', r2_pose[X])
    print('\t Y: ', r2_pose[Y])
    print('\t YAW: ', r2_pose[YAW])

    print()
    print()
    print()



  def get_pose(self, robot):
    a = 'occupancy_grid'
    b = robot + '/base_link'
    if self._tf.frameExists(a) and self._tf.frameExists(b):
      try:
        t = rospy.Time(0)
        position, orientation = self._tf.lookupTransform('/' + a, '/' + b, t)
        pose = np.zeros(3, dtype=np.float32)
        pose[X] = position[X]
        pose[Y] = position[Y]
        _, _, pose[YAW] = euler_from_quaternion(orientation)
        return pose

      except Exception as e:
        print(e)
    else:
      print('Unable to find:', self._tf.frameExists(a), self._tf.frameExists(b))


  @property
  def ready(self):
    return self._occupancy_grid is not None and not np.isnan(self._pose[0])

  @property
  def pose(self):
    return self._pose

  @property
  def occupancy_grid(self):
    return self._occupancy_grid

def run():
    rospy.init_node('follower_1_controller')
    l_publisher = rospy.Publisher('/' + LEADER + '/cmd_vel', Twist, queue_size=5)
    f1_publisher = rospy.Publisher('/' + FOLLOWER_1 + '/cmd_vel', Twist, queue_size=5)
    f2_publisher = rospy.Publisher('/' + FOLLOWER_2 + '/cmd_vel', Twist, queue_size=5)
    slam = SLAM()
    rate_limiter = rospy.Rate(100)

    # Stop moving message.
    stop_msg = Twist()
    stop_msg.linear.x = 0.
    stop_msg.angular.z = 0.

    previous_time = rospy.Time.now().to_sec()

    # Make sure the robot is stopped.
    i = 0
    while i < 10 and not rospy.is_shutdown():
      l_publisher.publish(stop_msg)
      f1_publisher.publish(stop_msg)
      f2_publisher.publish(stop_msg)
      rate_limiter.sleep()
      i += 1

    while not rospy.is_shutdown():
      slam.update()

      vel_msg = Twist()
      vel_msg.linear.x = 0.05
      vel_msg.linear.z = 0.02
      l_publisher.publish(vel_msg)
      f1_publisher.publish(vel_msg)
      f2_publisher.publish(vel_msg)

      rate_limiter.sleep()

if __name__ == '__main__':
    run()
